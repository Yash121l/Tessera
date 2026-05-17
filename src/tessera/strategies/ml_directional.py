"""ML Directional Strategy.

On each bar:
  (a) Maintain a rolling bar buffer per symbol (deque of recent bars).
  (b) Compute lightweight features (log returns, realised vol, Parkinson vol).
  (c) Get primary signal {-1, 0, +1} from the primary model.
  (d) Get meta-probability from the meta model (scale confidence).
  (e) Compute target_qty = sign * kelly_size * vol_target_scalar.
  (f) Submit a post-only LIMIT order; fall back to MARKET after taker_delay_bars
      if unfilled.

Signal delay (latency ablation):
  If signal_delay_bars > 0, the signal computed on bar t is acted upon on bar
  t + signal_delay_bars. A deque of pending signals is maintained per symbol.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId

from tessera.backtest.slippage import OHLCVSlippageModel
from tessera.strategies.base import TesseraBaseStrategy, TesseraStrategyConfig

logger = structlog.get_logger(__name__)

_SECS_PER_YEAR = 365.25 * 24 * 3600
_NS_PER_SEC = 1_000_000_000

# Minimum volatility to avoid division-by-zero in Kelly sizing
_MIN_VOL = 0.001
# Minimum number of bars in buffer before generating a signal
_MIN_HISTORY = 60


class MLDirectionalConfig(TesseraStrategyConfig, frozen=True):
    """Configuration for MLDirectionalStrategy."""

    primary_model_path: str = ""  # Path to primary LightGBM model dir
    meta_model_path: str = ""  # Empty string = no meta model
    feature_lookback: int = 300  # bars to keep in buffer
    adv_lookback: int = 1440  # bars used to estimate ADV for slippage
    slippage_k: float = 1.0
    half_spread_bps: float = 2.5
    min_trade_notional: float = 10.0  # USD; below this, skip rebalance


class MLDirectionalStrategy(TesseraBaseStrategy):
    """ML Directional alpha strategy.

    Requires a trained primary model. Meta model is optional. When no model
    path is given (e.g. in tests), falls back to a zero-signal (flat) mode.
    """

    def __init__(self, config: MLDirectionalConfig) -> None:
        super().__init__(config)
        self._ml_cfg = config
        self._primary_model: Any = None
        self._meta_model: Any = None
        # bar buffers: instrument_id str → deque[dict]
        self._bar_buffers: dict[str, deque[dict[str, float]]] = {}
        # pending signals: instrument_id str → deque[(signal, bar_count_remaining)]
        self._delayed_signals: dict[str, deque[tuple[int, float, int]]] = {}
        self._bar_counter: dict[str, int] = {}
        # per-symbol pending limit order id (str)
        self._pending_limit: dict[str, str | None] = {}
        self._pending_limit_bars: dict[str, int] = {}
        self._slippage = OHLCVSlippageModel(
            k=config.slippage_k,
            half_spread_bps=config.half_spread_bps,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        super().on_start()
        for id_str in self._cfg.instrument_ids:
            lookback = self._ml_cfg.feature_lookback
            self._bar_buffers[id_str] = deque(maxlen=lookback)
            self._delayed_signals[id_str] = deque()
            self._bar_counter[id_str] = 0
            self._pending_limit[id_str] = None
            self._pending_limit_bars[id_str] = 0

        if self._ml_cfg.primary_model_path:
            self._load_models()

    def _load_models(self) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel

        primary_path = Path(self._ml_cfg.primary_model_path)
        if primary_path.exists():
            self._primary_model = PrimaryLightGBMModel.load(primary_path)
            logger.info("primary_model_loaded", path=str(primary_path))

        if self._ml_cfg.meta_model_path:
            meta_path = Path(self._ml_cfg.meta_model_path)
            if meta_path.exists():
                from tessera.models.lightgbm_model import PrimaryLightGBMModel as MetaLGBM

                self._meta_model = MetaLGBM.load(meta_path)
                logger.info("meta_model_loaded", path=str(meta_path))

    # ------------------------------------------------------------------
    # Bar handler
    # ------------------------------------------------------------------

    def _on_bar_impl(self, bar: Bar) -> None:
        id_str = str(bar.bar_type.instrument_id)
        instr_id = bar.bar_type.instrument_id

        # Accumulate bar into buffer
        self._bar_buffers[id_str].append(
            {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "ts": bar.ts_event,
            }
        )
        self._bar_counter[id_str] = self._bar_counter.get(id_str, 0) + 1

        # Check if a delayed signal is ready to execute
        self._process_delayed_signals(id_str, instr_id, bar)

        # Check if pending limit needs to be cancelled + retried as market
        self._check_taker_fallback(id_str, instr_id, bar)

        if len(self._bar_buffers[id_str]) < _MIN_HISTORY:
            return

        # Compute features and get signal
        signal = self._compute_signal(id_str)
        meta_prob = self._compute_meta_prob(id_str, signal)

        if self._cfg.signal_delay_bars == 0:
            # Act immediately
            self._act_on_signal(signal, meta_prob, instr_id, bar)
        else:
            # Queue for later
            self._delayed_signals[id_str].append((signal, meta_prob, self._cfg.signal_delay_bars))

    def _process_delayed_signals(self, id_str: str, instr_id: InstrumentId, bar: Bar) -> None:
        if not self._delayed_signals[id_str]:
            return

        # Decrement counters; execute any that have reached 0
        updated: deque[tuple[int, float, int]] = deque()
        for signal, meta_prob, bars_left in self._delayed_signals[id_str]:
            bars_left -= 1
            if bars_left <= 0:
                self._act_on_signal(signal, meta_prob, instr_id, bar)
            else:
                updated.append((signal, meta_prob, bars_left))
        self._delayed_signals[id_str] = updated

    def _check_taker_fallback(self, id_str: str, instr_id: InstrumentId, bar: Bar) -> None:
        pending_oid = self._pending_limit.get(id_str)
        if pending_oid is None:
            return

        self._pending_limit_bars[id_str] = self._pending_limit_bars.get(id_str, 0) + 1
        if self._pending_limit_bars[id_str] >= self._cfg.taker_delay_bars:
            # Cancel the pending limit and submit a market order instead
            self.cancel_all_orders(instrument_id=instr_id)
            self._pending_limit[id_str] = None
            self._pending_limit_bars[id_str] = 0

    # ------------------------------------------------------------------
    # Signal + sizing
    # ------------------------------------------------------------------

    def _compute_signal(self, id_str: str) -> int:
        """Primary model prediction {-1, 0, +1}, or 0 if no model."""
        if self._primary_model is None:
            return 0

        features = self._build_features(id_str)
        if features is None:
            return 0

        pred = self._primary_model.predict(features)
        return int(pred[-1])

    def _compute_meta_prob(self, id_str: str, signal: int) -> float:
        """Meta-model probability in [0, 1]; defaults to 1.0 (full confidence)."""
        if self._meta_model is None or signal == 0:
            return 1.0

        features = self._build_features(id_str)
        if features is None:
            return 0.5

        proba = self._meta_model.predict_proba(features)
        # proba shape (1, n_classes); take max class prob as confidence
        return float(proba[-1].max())

    def _build_features(self, id_str: str) -> pd.DataFrame | None:
        buf = list(self._bar_buffers[id_str])
        if len(buf) < _MIN_HISTORY:
            return None

        closes = np.array([b["close"] for b in buf])
        highs = np.array([b["high"] for b in buf])
        lows = np.array([b["low"] for b in buf])

        log_ret = np.diff(np.log(closes))
        if len(log_ret) < 60:
            return None

        rv_60 = float(log_ret[-60:].std()) * math.sqrt(1440)
        rv_300 = float(log_ret[-min(300, len(log_ret)) :].std()) * math.sqrt(1440)
        pk_60 = _parkinson_vol(highs[-60:], lows[-60:])

        row = {
            "log_return_1": float(log_ret[-1]),
            "log_return_5": float(np.sum(log_ret[-5:])),
            "log_return_60": float(np.sum(log_ret[-60:])),
            "realized_vol_60": rv_60,
            "realized_vol_300": rv_300,
            "parkinson_vol_60": pk_60,
        }
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _act_on_signal(
        self, signal: int, meta_prob: float, instr_id: InstrumentId, bar: Bar
    ) -> None:
        id_str = str(instr_id)
        close = float(bar.close)
        instr = self.cache.instrument(instr_id)
        if instr is None:
            return

        target_qty = self._kelly_target_qty(signal, meta_prob, id_str, close)
        current_qty = self._net_position(instr_id)
        delta = target_qty - current_qty

        if abs(delta * close) < self._ml_cfg.min_trade_notional:
            return  # Not worth rebalancing

        # Apply slippage to the limit price
        adv = self._estimate_adv_notional(id_str)
        order_notional = abs(delta) * close
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        slipped_price = self._slippage.adjust_price(
            close, side.name.lower(), order_notional, adv, id_str, is_taker=False
        )

        qty = instr.make_qty(abs(delta))
        limit_price = instr.make_price(slipped_price)

        order = self.order_factory.limit(
            instrument_id=instr_id,
            order_side=side,
            quantity=qty,
            price=limit_price,
            time_in_force=TimeInForce.GTC,
            post_only=True,
        )
        self.submit_order(order)
        self._pending_limit[id_str] = str(order.client_order_id)
        self._pending_limit_bars[id_str] = 0
        logger.debug(
            "order_submitted",
            instrument=id_str,
            side=side.name,
            qty=float(qty),
            price=float(limit_price),
            signal=signal,
            meta_prob=round(meta_prob, 3),
        )

    def _kelly_target_qty(self, signal: int, meta_prob: float, id_str: str, price: float) -> float:
        """Target position quantity using fractional Kelly + vol targeting."""
        if signal == 0:
            return 0.0

        buf = list(self._bar_buffers[id_str])
        closes = np.array([b["close"] for b in buf[-300:]])
        if len(closes) < 2:
            return 0.0

        log_ret = np.diff(np.log(closes))
        bar_vol = max(float(log_ret.std()), _MIN_VOL / math.sqrt(1440))
        annual_vol = bar_vol * math.sqrt(1440)

        # Vol target scalar: shrinks position when vol is high
        vol_scalar = min(self._cfg.vol_target_pct / annual_vol, 3.0)

        # Confidence from meta model: map meta_prob ∈ [0,1] → confidence ∈ [-1,1]
        confidence = 2.0 * meta_prob - 1.0  # [0,1] → [-1,1]

        # Portfolio equity estimate
        venues = [InstrumentId.from_str(s).venue for s in self._cfg.instrument_ids]
        equity = sum(self._current_equity(str(v)) for v in set(venues)) or 100_000.0

        max_notional = equity * self._cfg.max_position_pct
        kelly_notional = self._cfg.kelly_fraction * confidence * vol_scalar * equity
        notional = min(abs(kelly_notional), max_notional)
        qty = notional / price
        return signal * qty

    def _estimate_adv_notional(self, id_str: str) -> float:
        buf = list(self._bar_buffers[id_str])
        lookback = min(self._ml_cfg.adv_lookback, len(buf))
        if lookback < 1:
            return 1_000_000.0
        vols = [b["volume"] * b["close"] for b in buf[-lookback:]]
        # Daily volume: sum of per-bar volume (bars are 1-minute → 1440 bars/day)
        bars_per_day = 1440
        daily_vol = sum(vols) / max(1, lookback) * bars_per_day
        return max(daily_vol, 1_000.0)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _parkinson_vol(highs: np.ndarray, lows: np.ndarray) -> float:
    """Parkinson's (high-low range) volatility estimator, annualised."""
    log_hl = np.log(highs / lows)
    pk_bar = float(np.mean(log_hl**2) / (4 * math.log(2)))
    return math.sqrt(pk_bar * 1440)
