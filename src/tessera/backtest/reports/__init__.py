"""Tessera reporting layer: statistical evaluation and tearsheet generation."""

from __future__ import annotations

from tessera.backtest.reports.bootstrap import block_bootstrap_sharpe
from tessera.backtest.reports.deflated_sharpe import compute_trial_count, deflated_sharpe
from tessera.backtest.reports.probabilistic_sharpe import probabilistic_sharpe
from tessera.backtest.reports.stress import STRESS_WINDOWS, compute_stress_pnls
from tessera.backtest.reports.tearsheet import generate_tearsheet

__all__ = [
    "deflated_sharpe",
    "probabilistic_sharpe",
    "block_bootstrap_sharpe",
    "compute_trial_count",
    "compute_stress_pnls",
    "generate_tearsheet",
    "STRESS_WINDOWS",
]
