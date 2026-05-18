"""Unit tests for DSR/PSR implementations.

Key invariants:
  - DSR is strictly less than PSR with 1 trial (more trials = higher bar).
  - DSR with SR=2.0 and 100 trials is measurably below the single-trial PSR.
  - PSR(SR=0) ≈ 0.5 (observed SR equals benchmark → 50/50).
  - PSR increases monotonically with SR.
  - compute_trial_count correctly sums Optuna + manual counts.
"""

from __future__ import annotations

import math

from tessera.backtest.reports.deflated_sharpe import compute_trial_count, deflated_sharpe
from tessera.backtest.reports.probabilistic_sharpe import probabilistic_sharpe

# ---------------------------------------------------------------------------
# PSR tests
# ---------------------------------------------------------------------------


def test_psr_at_benchmark_is_half():
    """PSR(observed=0, benchmark=0) should be exactly 0.5."""
    psr = probabilistic_sharpe(0.0, 0.0, n_obs=252)
    assert abs(psr - 0.5) < 1e-6


def test_psr_increases_with_sr():
    """Higher observed SR → higher PSR (monotonic with fixed benchmark)."""
    srs = [0.5, 1.0, 1.5, 2.0, 3.0]
    psrs = [probabilistic_sharpe(sr, 0.0, n_obs=252) for sr in srs]
    for i in range(len(psrs) - 1):
        assert psrs[i] < psrs[i + 1], f"PSR not monotone at SR={srs[i]}"


def test_psr_increases_with_n_obs():
    """More observations → narrower uncertainty → higher PSR for SR > 0."""
    psr_small = probabilistic_sharpe(1.5, 0.0, n_obs=100)
    psr_large = probabilistic_sharpe(1.5, 0.0, n_obs=1000)
    assert psr_large > psr_small


def test_psr_is_probability():
    """PSR output must be in [0, 1]."""
    for sr in [-5.0, -1.0, 0.0, 1.0, 2.5, 10.0]:
        p = probabilistic_sharpe(sr, 0.0, n_obs=504)
        assert 0.0 <= p <= 1.0, f"PSR={p} out of range for SR={sr}"


def test_psr_skew_kurt_correction():
    """Non-normal returns should reduce PSR (wider variance → less confidence)."""
    psr_normal = probabilistic_sharpe(2.0, 0.0, n_obs=252, skew=0.0, kurt=0.0)
    # Negative skew and positive excess kurtosis both inflate variance
    psr_fat_tail = probabilistic_sharpe(2.0, 0.0, n_obs=252, skew=-1.0, kurt=3.0)
    assert psr_fat_tail < psr_normal, "Fat-tail returns should reduce PSR"


# ---------------------------------------------------------------------------
# DSR tests — the primary spec requirement
# ---------------------------------------------------------------------------


def test_dsr_below_single_trial_psr():
    """DSR with 100 trials must be strictly less than PSR with 1 trial."""
    observed_sr = 2.0
    sr_std = 1.0
    n_obs = 252

    psr_single = deflated_sharpe(observed_sr, sr_std, n_trials=1, n_obs=n_obs)
    dsr_100 = deflated_sharpe(observed_sr, sr_std, n_trials=100, n_obs=n_obs)

    assert dsr_100 < psr_single, f"DSR({dsr_100:.4f}) should be < PSR_single({psr_single:.4f})"


def test_dsr_measurably_below_2():
    """DSR with SR=2, 100 trials, n_obs=252 should be well below 1.0 (≤ 0.5)."""
    dsr = deflated_sharpe(observed_sr=2.0, sr_std=1.0, n_trials=100, n_obs=252)
    # SR=2 with 100 trials: expected max ≈ 2.53 → DSR ≈ 0.31
    assert dsr < 0.5, f"DSR={dsr:.4f} should be < 0.5 for SR=2 with 100 trials"
    assert not math.isnan(dsr)


def test_dsr_increases_with_sr():
    """Holding n_trials constant, higher observed SR → higher DSR."""
    srs = [1.0, 2.0, 3.0, 4.0, 5.0]
    dsrs = [deflated_sharpe(sr, 1.0, n_trials=100, n_obs=252) for sr in srs]
    for i in range(len(dsrs) - 1):
        assert dsrs[i] < dsrs[i + 1], f"DSR not monotone at SR={srs[i]}"


def test_dsr_decreases_with_trials():
    """More trials → higher penalty → lower DSR (all else equal)."""
    dsr_10 = deflated_sharpe(3.0, 1.0, n_trials=10, n_obs=252)
    dsr_100 = deflated_sharpe(3.0, 1.0, n_trials=100, n_obs=252)
    dsr_1000 = deflated_sharpe(3.0, 1.0, n_trials=1000, n_obs=252)
    msg = f"DSR should decrease: {dsr_10:.4f} > {dsr_100:.4f} > {dsr_1000:.4f}"
    assert dsr_10 > dsr_100 > dsr_1000, msg


def test_dsr_high_sr_passes():
    """SR=5 with 100 trials and 5 years of daily data should have DSR > 0.95."""
    dsr = deflated_sharpe(observed_sr=5.0, sr_std=1.0, n_trials=100, n_obs=252 * 5)
    assert dsr > 0.95, f"DSR={dsr:.4f} should be > 0.95 for SR=5 with 5 years"


def test_dsr_is_probability():
    """DSR output must be in [0, 1]."""
    for sr in [0.1, 1.0, 2.0, 5.0, 10.0]:
        d = deflated_sharpe(sr, 1.0, n_trials=100, n_obs=252)
        assert 0.0 <= d <= 1.0, f"DSR={d} out of range for SR={sr}"


def test_dsr_1_trial_equals_psr():
    """With n_trials=1, DSR should equal PSR vs. benchmark 0."""
    sr = 2.5
    n_obs = 500
    psr = probabilistic_sharpe(sr, 0.0, n_obs=n_obs)
    dsr = deflated_sharpe(sr, 1.0, n_trials=1, n_obs=n_obs)
    assert abs(psr - dsr) < 1e-6, f"PSR={psr:.6f} != DSR_1trial={dsr:.6f}"


# ---------------------------------------------------------------------------
# compute_trial_count tests
# ---------------------------------------------------------------------------


def test_trial_count_manual_int():
    assert compute_trial_count(manual_configs=40) == 40


def test_trial_count_manual_list():
    configs = [{"lr": 0.01}, {"lr": 0.05}, {"lr": 0.1}]
    assert compute_trial_count(manual_configs=configs) == 3


def test_trial_count_combined():
    """Optuna + manual should sum correctly."""

    class FakeStudy:
        trials = list(range(100))

    assert compute_trial_count(FakeStudy(), manual_configs=40) == 140


def test_trial_count_minimum_1():
    """No input should return 1, not 0."""
    assert compute_trial_count() >= 1


def test_trial_count_none_inputs():
    assert compute_trial_count(optuna_study=None, manual_configs=None) >= 1
