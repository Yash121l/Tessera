"""Backtest smoke test — exercises the full pipeline on fixture data.

Pipeline:
  synthetic bars → lightweight features → triple-barrier labels
  → LightGBM fit (n_estimators=10, no Optuna)
  → TesseraBacktestEngine.from_bars()
  → tearsheet HTML generation

Budget: < 90 seconds on GitHub Actions ubuntu-latest.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

_HAS_NAUTILUS = False
_HAS_LGB = False
try:
    import nautilus_trader  # noqa: F401

    _HAS_NAUTILUS = True
except ImportError:
    pass

try:
    import lightgbm  # noqa: F401

    _HAS_LGB = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not (_HAS_NAUTILUS and _HAS_LGB),
    reason="nautilus-trader or lightgbm not installed",
)

# ---------------------------------------------------------------------------
# Helpers — synthetic bar construction
# ---------------------------------------------------------------------------

SEED = 42
N_BARS = 200
# ETH-USDT-PERP.BINANCE: price_precision=2, matches our Price(v, precision=2) bars.
SYMBOL_ID = "ETH-USDT-PERP.BINANCE"
SIGMA = 0.001  # per-bar log-return std
S0 = 3_000.0
_NS_PER_MIN = 60 * 1_000_000_000


def _make_ohlcv_df(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """Deterministic GBM OHLCV as a DataFrame (1-min bars)."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, SIGMA, n)
    close = S0 * np.exp(np.cumsum(log_ret))
    noise_h = rng.uniform(0.0, 2 * SIGMA, n)
    noise_l = rng.uniform(0.0, 2 * SIGMA, n)
    high = close * (1.0 + noise_h)
    low = close * (1.0 - noise_l)
    open_ = np.empty(n)
    open_[0] = S0
    open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.uniform(50.0, 500.0, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=ts,
    )


def _df_to_nautilus_bars(df: pd.DataFrame, symbol_id: str):  # type: ignore[return]
    from nautilus_trader.model.data import Bar, BarSpecification, BarType
    from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.objects import Price, Quantity

    parts = symbol_id.split(".")
    sym = Symbol(parts[0])
    venue = Venue(parts[1])
    bar_type = BarType(
        instrument_id=InstrumentId(sym, venue),
        bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )

    # ETH instrument spec: price_precision=2, size_precision=2
    def _p(v: float) -> Price:
        return Price(round(v, 2), 2)

    def _q(v: float) -> Quantity:
        return Quantity(round(v, 2), 2)

    ts_base = int(df.index[0].timestamp() * 1e9)
    bars = []
    for i, row in enumerate(df.itertuples()):
        ts = ts_base + i * _NS_PER_MIN
        bars.append(
            Bar(
                bar_type=bar_type,
                open=_p(row.open),
                high=_p(row.high),
                low=_p(row.low),
                close=_p(row.close),
                volume=_q(row.volume),
                ts_event=ts,
                ts_init=ts,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# Helpers — feature extraction (matches MLDirectionalStrategy._build_features)
# ---------------------------------------------------------------------------


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the same features that MLDirectionalStrategy computes per bar."""
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    log_ret = np.diff(np.log(closes))
    rows = []
    for i in range(60, len(log_ret)):
        rv_60 = float(log_ret[i - 60 : i].std()) * math.sqrt(1440)
        rv_300 = float(log_ret[max(0, i - 300) : i].std()) * math.sqrt(1440)
        log_hl = np.log(highs[i - 60 : i] / lows[i - 60 : i])
        pk = math.sqrt(float(np.mean(log_hl**2) / (4 * math.log(2))) * 1440)
        rows.append(
            {
                "log_return_1": float(log_ret[i]),
                "log_return_5": float(np.sum(log_ret[i - 5 : i])),
                "log_return_60": float(np.sum(log_ret[i - 60 : i])),
                "realized_vol_60": rv_60,
                "realized_vol_300": rv_300,
                "parkinson_vol_60": pk,
            }
        )
    return pd.DataFrame(rows)


def _make_labels(df: pd.DataFrame) -> np.ndarray:
    """Sign-of-return labels for smoke testing.

    ~50% +1, ~50% -1, with a single forced 0 to satisfy LightGBM num_class=3.
    This ensures the trained model predicts ±1 frequently, producing actual
    trades in the backtest engine (unlike a high-neutral-fraction labeling scheme
    where a 10-estimator model just learns to predict 0 on all small-N inputs).
    """
    closes = df["close"].values
    log_ret = np.diff(np.log(closes))
    labels = np.sign(log_ret[60:]).astype(int)
    # Ensure all 3 classes present (LightGBM num_class=3 requires this)
    if not (labels == 0).any():
        labels[0] = 0
    return labels


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_backtest_smoke():
    """Full pipeline smoke test: features → LightGBM → BacktestEngine → tearsheet."""
    t0 = time.monotonic()

    from tessera.backtest.engine import TesseraBacktestEngine
    from tessera.backtest.reports.tearsheet import generate_tearsheet
    from tessera.models.lightgbm_model import PrimaryLightGBMModel
    from tessera.strategies.ml_directional import MLDirectionalConfig, MLDirectionalStrategy

    # 1. Generate synthetic OHLCV
    df = _make_ohlcv_df(N_BARS, SEED)

    # 2. Extract features + labels
    X = _extract_features(df)  # noqa: N806
    y_raw = _make_labels(df)
    n = min(len(X), len(y_raw))
    X = X.iloc[:n].reset_index(drop=True)  # noqa: N806
    y = pd.Series(y_raw[:n], name="label")

    assert len(X) >= 30, "Not enough bars for training"

    # 3. Fit minimal LightGBM (10 estimators, no Optuna)
    with tempfile.TemporaryDirectory() as tmpdir:
        # num_class=3 required by LightGBM multiclass for {-1, 0, +1} labels.
        model = PrimaryLightGBMModel(seed=SEED, n_estimators=10, num_class=3)
        model.fit(X, y)
        model_path = Path(tmpdir) / "primary"
        model.save(model_path)

        # 4. Build strategy with saved model
        cfg = MLDirectionalConfig(
            instrument_ids=[SYMBOL_ID],
            primary_model_path=str(model_path),
            min_trade_notional=1.0,  # low threshold → more trades on small fixture
            kelly_fraction=0.25,
            vol_target_pct=0.15,
            max_position_pct=0.20,
            # Synthetic bars have zero spread: POST_ONLY limits would be rejected
            # as marketable. Use non-post-only limits for this smoke test.
            post_only_orders=False,
        )
        strategy = MLDirectionalStrategy(cfg)

        # 5. Convert bars and run engine
        bars = _df_to_nautilus_bars(df, SYMBOL_ID)
        engine = TesseraBacktestEngine.from_bars(
            bars_by_symbol={SYMBOL_ID: bars},
            strategy=strategy,
            run_id="smoke_test",
            seed=SEED,
        )
        result = engine.run()

        # 6. Core assertions
        assert result.n_trades >= 1, (
            f"Expected ≥1 trade in smoke run; got {result.n_trades}. "
            "Strategy may be stuck in flat mode."
        )
        assert math.isfinite(result.sharpe_ratio), f"Sharpe is not finite: {result.sharpe_ratio}"
        assert not math.isnan(result.sharpe_ratio), "Sharpe is NaN"

        # 7. Tearsheet generation — must contain "Deflated SR"
        tearsheet_path = Path(tmpdir) / "tearsheet.html"

        # generate_tearsheet expects a pd.Series of returns, not a BacktestResult.
        # The smoke test spans ~3 hours (200 one-minute bars), so daily-resampling
        # the equity curve produces a single data point.  When that happens we fall
        # back to a minimal synthetic return series so QuantStats can render HTML.
        eq = result.equity_curve
        daily_returns = eq.pct_change().dropna()
        if len(daily_returns) < 3:
            daily_returns = pd.Series(
                [0.001, -0.0005, 0.002],
                index=pd.date_range("2024-01-01", periods=3, freq="1D"),
                name="returns",
            )

        generate_tearsheet(
            daily_returns,
            None,  # benchmark_returns
            output_path=tearsheet_path,
            trial_count=10,
        )
        assert tearsheet_path.exists(), "Tearsheet file was not written"
        html_text = tearsheet_path.read_text(encoding="utf-8", errors="replace")
        assert "Deflated SR" in html_text, "Tearsheet HTML does not contain 'Deflated SR'"

        elapsed = time.monotonic() - t0
        assert elapsed < 90.0, f"Smoke test took {elapsed:.1f}s (budget: 90s)"
