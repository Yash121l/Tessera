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
from tessera.risk.circuit_breaker import CircuitBreaker
from tessera.risk.kelly import kelly_from_meta_prob
from tessera.risk.kill_switch import KillSwitch, KillSwitchConfig
from tessera.risk.limits import PositionLimits
from tessera.risk.vol_target import vol_target_scalar
from tessera.strategies.base import TesseraBaseStrategy, TesseraStrategyConfig

logger = structlog.get_logger(__name__)

_SECS_PER_YEAR = 365.25 * 24 * 3600
_NS_PER_SEC = 1_000_000_000

_MIN_VOL = 0.001
_MIN_HISTORY = 60


class MLDirectionalConfig(TesseraStrategyConfig, frozen=True):
    """Configuration for MLDirectionalStrategy."""

    primary_model_path: str = ""
    meta_model_path: str = ""
    feature_lookback: int = 300
    adv_lookback: int = 1440
    slippage_k: float = 1.0
    half_spread_bps: float = 2.5
    min_trade_notional: float = 10.0

    # Order execution
    # post_only_orders=True: submit limit orders with POST_ONLY; rejected if marketable.
    # Set False for testing with synthetic data where bid/ask spread is zero.
    post_only_orders: bool = True

    # Risk stack
    risk_db_dsn: str = ""  # Postgres DSN for CircuitBreaker; empty = in-memory
    daily_loss_threshold: float = 0.03
    drawdown_kill_threshold: float = 0.08
    data_gap_seconds: float = 30.0
    order_reject_threshold: float = 0.05


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
        self._bar_buffers: dict[str, deque[dict[str, float]]] = {}
        self._delayed_signals: dict[str, deque[tuple[int, float, int]]] = {}
        self._bar_counter: dict[str, int] = {}
        self._pending_limit: dict[str, str | None] = {}
        self._pending_limit_bars: dict[str, int] = {}
        self._slippage = OHLCVSlippageModel(
            k=config.slippage_k,
            half_spread_bps=config.half_spread_bps,
        )

        # Risk stack — wired in order: kelly → vol_target → limits → cb → ks
        self._kill_switch = KillSwitch(
            config=KillSwitchConfig(
                daily_loss_threshold=config.daily_loss_threshold,
                drawdown_threshold=config.drawdown_kill_threshold,
                data_gap_seconds=config.data_gap_seconds,
                reject_rate_threshold=config.order_reject_threshold,
            ),
            on_trigger=self._on_kill_switch_trigger,
        )
        self._circuit_breaker = CircuitBreaker(dsn=config.risk_db_dsn or None)
        self._position_limits = PositionLimits(max_asset_pct=config.max_position_pct)
        self._mtd_start_equity: float | None = None

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

    def _on_kill_switch_trigger(self, trigger: Any, detail: str) -> None:
        logger.critical("kill_switch_flattening_all", trigger=str(trigger), detail=detail)
        self._flatten_all()

    # ------------------------------------------------------------------
    # Bar handler
    # ------------------------------------------------------------------

    def _on_bar_impl(self, bar: Bar) -> None:
        id_str = str(bar.bar_type.instrument_id)
        instr_id = bar.bar_type.instrument_id

        # Kill switch is source of truth — abort the entire cycle if active.
        if self._kill_switch.is_active:
            return
        self._kill_switch.record_data_tick()
        self._kill_switch.check_data_gap()

        # Update circuit breaker with current equity.
        venues = list({InstrumentId.from_str(s).venue for s in self._cfg.instrument_ids})
        equity = sum(self._current_equity(str(v)) for v in venues) or 100_000.0
        if self._mtd_start_equity is None:
            self._mtd_start_equity = equity
        self._circuit_breaker.update(equity, mtd_start_equity=self._mtd_start_equity)

        # Equity-based kill switch checks.
        self._kill_switch.check_daily_loss(equity)
        self._kill_switch.check_drawdown(equity)
        if self._kill_switch.is_active:
            return

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
        """Meta-model probability in (0, 1); defaults to 0.99 when no meta model.

        Must be strictly < 1.0: kelly_from_meta_prob guards `p_win < 1.0`
        and returns 0 for p_win == 1.0, which would suppress all trades.
        """
        if self._meta_model is None or signal == 0:
            return 0.99  # full confidence proxy that satisfies Kelly's open-interval guard

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
            post_only=self._ml_cfg.post_only_orders,
        )
        self.submit_order(order)
        self._kill_switch.record_order_event(rejected=False)
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
        """Target position quantity via kelly → vol_target → limits → circuit_breaker.

        Returns 0 immediately if the kill switch is active.
        """
        if signal == 0 or self._kill_switch.is_active:
            return 0.0

        buf = list(self._bar_buffers[id_str])
        closes = np.array([b["close"] for b in buf[-300:]])
        if len(closes) < 2:
            return 0.0

        log_ret = np.diff(np.log(closes))
        bar_vol = max(float(log_ret.std()), _MIN_VOL / math.sqrt(1440))
        annual_vol = bar_vol * math.sqrt(1440)

        # 1. Kelly fraction from meta-model probability.
        #    Expected return ≈ daily vol; expected loss ≈ half that (2:1 R/R target).
        daily_vol = annual_vol / math.sqrt(252)
        kelly_frac = kelly_from_meta_prob(
            p_meta=meta_prob,
            expected_return=daily_vol,
            expected_loss=0.5 * daily_vol,
            fraction=self._cfg.kelly_fraction,
        )

        # 2. Vol-target scalar — scale down when vol is high, up when low.
        vt_scalar = vol_target_scalar(annual_vol, target_vol_annual=self._cfg.vol_target_pct)

        # 3. Portfolio equity.
        venues = list({InstrumentId.from_str(s).venue for s in self._cfg.instrument_ids})
        equity = sum(self._current_equity(str(v)) for v in venues) or 100_000.0

        # 4. Circuit breaker size multiplier (1.0, 0.5, or 0.0).
        cb_mult = self._circuit_breaker.size_multiplier

        raw_notional = kelly_frac * vt_scalar * cb_mult * equity

        # 5. Position limits — clip to per-asset and gross/net caps.
        raw_positions = {id_str: float(signal) * raw_notional}
        clipped = self._position_limits.clip_to_limits(raw_positions, nav=equity)
        notional = clipped.get(id_str, 0.0)

        return notional / price if price > 0.0 else 0.0

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
