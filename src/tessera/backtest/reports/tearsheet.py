"""Generate an enhanced QuantStats HTML tearsheet.

Overlays the following metrics that QuantStats omits:
  - Deflated Sharpe Ratio (DSR) with trial count
  - Probabilistic Sharpe Ratio (PSR) vs. zero benchmark
  - Bootstrap 95% CI for the annualized Sharpe
  - Stress-window PnL table
  - Benchmark (BTC buy-and-hold) comparison when provided

The QuantStats HTML is generated first, then our panel is injected
immediately after <body> so it renders at the top of the report.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    output_path: str | Path,
    trial_count: int = 1,
    manual_metrics_to_overlay: dict[str, Any] | None = None,
    sr_std: float | None = None,
    n_obs_per_year: int = 252,
    block_size: int | None = None,
    n_bootstrap: int = 10_000,
    test_start_date: str | None = None,
    title: str = "Tessera Strategy Tearsheet",
) -> Path:
    """Generate a QuantStats HTML report with DSR/PSR/bootstrap overlays.

    Args:
        returns: Strategy daily return series with a DatetimeIndex.
        benchmark_returns: Buy-and-hold benchmark returns (same index).
            Pass None to omit the benchmark.
        output_path: Where to write the HTML file.
        trial_count: Total number of independent strategies / configs tested
            (Optuna trials + manual experiments).  See compute_trial_count().
        manual_metrics_to_overlay: Dict of pre-computed metrics to inject
            without recomputation (e.g. {'deflated_sharpe': 0.72}).
        sr_std: Standard deviation of annualized SRs across the trial pool.
            Defaults to 1/√T (conservative).
        n_obs_per_year: Annualization factor (252 for daily).
        block_size: Block size for stationary bootstrap.  Defaults to ⌈√T⌉.
        n_bootstrap: Bootstrap resamples (default 10 000).
        test_start_date: ISO date of first OOS bar (for stress IS/OOS label).
        title: Report title shown in the HTML.

    Returns:
        Path to the written HTML file.
    """
    import matplotlib

    matplotlib.use("Agg")

    import quantstats as qs

    from tessera.backtest.reports.bootstrap import block_bootstrap_sharpe
    from tessera.backtest.reports.deflated_sharpe import deflated_sharpe
    from tessera.backtest.reports.probabilistic_sharpe import (
        probabilistic_sharpe,
        sharpe_skew_kurt,
    )
    from tessera.backtest.reports.stress import compute_stress_pnls

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = returns.dropna()
    n_obs = len(r)

    # --- Compute or extract overlay metrics ---
    overlay: dict[str, Any] = dict(manual_metrics_to_overlay or {})

    r_arr = np.asarray(r.values, dtype=float)
    skew, kurt = sharpe_skew_kurt(r_arr)

    if "point_sr" not in overlay:
        mu = float(r_arr.mean())
        sigma = float(r_arr.std(ddof=1)) if len(r_arr) > 1 else 1e-12
        overlay["point_sr"] = (
            float(mu / sigma * math.sqrt(n_obs_per_year)) if sigma > 1e-12 else 0.0
        )

    obs_sr = overlay["point_sr"]
    eff_sr_std = sr_std if sr_std is not None else (1.0 / math.sqrt(max(n_obs, 1)))

    if "psr" not in overlay:
        overlay["psr"] = probabilistic_sharpe(obs_sr, 0.0, n_obs, skew, kurt, n_obs_per_year)

    if "dsr" not in overlay:
        overlay["dsr"] = deflated_sharpe(
            obs_sr, eff_sr_std, trial_count, n_obs, skew, kurt, n_obs_per_year
        )

    if "bootstrap_lo" not in overlay or "bootstrap_hi" not in overlay:
        try:
            lo, _, hi = block_bootstrap_sharpe(
                r,
                block_size=block_size,
                n_resamples=n_bootstrap,
                annualization_factor=n_obs_per_year,
                seed=42,
            )
            overlay["bootstrap_lo"] = lo
            overlay["bootstrap_hi"] = hi
        except Exception:
            overlay["bootstrap_lo"] = float("nan")
            overlay["bootstrap_hi"] = float("nan")

    stress_df = compute_stress_pnls(
        r, test_start_date=test_start_date, annualization_factor=n_obs_per_year
    )

    # --- Generate QuantStats HTML ---
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        qs.reports.html(  # type: ignore[no-untyped-call]
            r,
            benchmark=benchmark_returns,
            output=str(tmp_path),
            title=title,
        )
        qs_html = tmp_path.read_text(encoding="utf-8", errors="replace")
    finally:
        tmp_path.unlink(missing_ok=True)

    # --- Build our custom panel ---
    panel_html = _build_overlay_panel(
        overlay, stress_df, trial_count, eff_sr_std, skew, kurt, n_obs
    )

    # Inject after <body> (or append if tag not found)
    inject_after = "<body>"
    idx = qs_html.find(inject_after)
    if idx >= 0:
        insert_at = idx + len(inject_after)
        combined = qs_html[:insert_at] + "\n" + panel_html + "\n" + qs_html[insert_at:]
    else:
        combined = qs_html + "\n" + panel_html

    output_path.write_text(combined, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# HTML panel builder
# ---------------------------------------------------------------------------

_CSS = """
<style>
.ts-panel {
  font-family: Arial, Helvetica, sans-serif;
  max-width: 960px;
  margin: 24px auto 0 auto;
  padding: 0 20px;
}
.ts-panel h2 {
  font-size: 18px;
  font-weight: 700;
  color: #333;
  border-bottom: 2px solid #e74c3c;
  padding-bottom: 6px;
  margin-bottom: 12px;
}
.ts-panel h3 {
  font-size: 14px;
  font-weight: 600;
  color: #555;
  margin: 16px 0 6px 0;
}
.ts-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-bottom: 18px;
}
.ts-table th {
  background: #f5f5f5;
  border: 1px solid #ddd;
  padding: 7px 12px;
  text-align: left;
  font-weight: 600;
  color: #444;
}
.ts-table td {
  border: 1px solid #eee;
  padding: 6px 12px;
  color: #333;
}
.ts-table tr:nth-child(even) { background: #fafafa; }
.ts-pass { color: #27ae60; font-weight: 700; }
.ts-warn { color: #e67e22; font-weight: 700; }
.ts-fail { color: #e74c3c; font-weight: 700; }
.ts-na   { color: #aaa; }
.ts-note {
  font-size: 11px;
  color: #888;
  margin-top: 4px;
}
</style>
"""


def _fmt_pct(v: float) -> str:
    if math.isnan(v):
        return '<span class="ts-na">n/a</span>'
    return f"{v * 100:.2f}%"


def _fmt_float(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return '<span class="ts-na">n/a</span>'
    return f"{v:.{decimals}f}"


def _traffic(v: float, thresholds: tuple[float, float]) -> str:
    if math.isnan(v):
        return "ts-na"
    lo, hi = thresholds
    if v >= hi:
        return "ts-pass"
    if v >= lo:
        return "ts-warn"
    return "ts-fail"


def _build_overlay_panel(
    overlay: dict[str, Any],
    stress_df: pd.DataFrame,
    trial_count: int,
    sr_std: float,
    skew: float,
    kurt: float,
    n_obs: int,
) -> str:
    dsr = overlay.get("dsr", float("nan"))
    psr = overlay.get("psr", float("nan"))
    pt_sr = overlay.get("point_sr", float("nan"))
    bs_lo = overlay.get("bootstrap_lo", float("nan"))
    bs_hi = overlay.get("bootstrap_hi", float("nan"))

    dsr_cls = _traffic(dsr, (0.75, 0.95))
    psr_cls = _traffic(psr, (0.75, 0.95))
    sr_cls = _traffic(pt_sr, (0.5, 1.5))

    # Main metrics table
    metrics_rows = f"""
    <tr>
      <td>Point-estimate SR (annualised)</td>
      <td class="{sr_cls}">{_fmt_float(pt_sr, 3)}</td>
      <td>√T-annualised μ/σ</td>
    </tr>
    <tr>
      <td>Bootstrap 95% CI for SR</td>
      <td>[{_fmt_float(bs_lo, 3)}, {_fmt_float(bs_hi, 3)}]</td>
      <td>Stationary block bootstrap (Politis &amp; Romano 1994)</td>
    </tr>
    <tr>
      <td>Probabilistic SR (PSR)</td>
      <td class="{psr_cls}">{_fmt_float(psr, 4)}</td>
      <td>P(true SR &gt; 0 | data), accounts for skew/kurt</td>
    </tr>
    <tr>
      <td>Deflated SR (DSR)</td>
      <td class="{dsr_cls}">{_fmt_float(dsr, 4)}</td>
      <td>PSR adjusted for {trial_count:,} independent trials</td>
    </tr>
    <tr>
      <td>Trial count (N)</td>
      <td>{trial_count:,}</td>
      <td>Optuna trials + manual notebook experiments</td>
    </tr>
    <tr>
      <td>SR std across trials (σ_SR)</td>
      <td>{_fmt_float(sr_std, 4)}</td>
      <td>Cross-trial SR dispersion used in DSR</td>
    </tr>
    <tr>
      <td>Return skewness / excess kurt</td>
      <td>{_fmt_float(skew, 3)} / {_fmt_float(kurt, 3)}</td>
      <td>Used in PSR/DSR variance correction</td>
    </tr>
    <tr>
      <td>Observations (T)</td>
      <td>{n_obs:,}</td>
      <td>Return bars used for SR estimation</td>
    </tr>
    """

    # Stress window table
    stress_rows = ""
    for _, row in stress_df.iterrows():
        cov_badge = (
            '<span class="ts-pass">full</span>'
            if row["coverage"] == "full"
            else (
                '<span class="ts-warn">partial</span>'
                if row["coverage"] == "partial"
                else '<span class="ts-na">none</span>'
            )
        )
        is_badge = (
            '<span class="ts-warn">IS</span>'
            if row["in_sample"] == "IS"
            else (
                '<span class="ts-pass">OOS</span>'
                if row["in_sample"] == "OOS"
                else '<span class="ts-na">unknown</span>'
            )
        )
        stress_rows += f"""
        <tr>
          <td>{row["event"]}</td>
          <td>{row["start"]} → {row["end"]}</td>
          <td>{is_badge}</td>
          <td>{cov_badge}</td>
          <td>{_fmt_pct(row["total_return"])}</td>
          <td>{_fmt_pct(row["max_drawdown"])}</td>
          <td>{_fmt_float(row["sharpe"], 2)}</td>
        </tr>"""

    dsr_interp = (
        "STRONG: DSR ≥ 0.95 — high confidence of genuine skill after multiple-testing penalty."
        if (not math.isnan(dsr) and dsr >= 0.95)
        else (
            "WEAK: DSR &lt; 0.95 — insufficient evidence of skill after {trial_count} trials.  "
            "Likely noise."
            if not math.isnan(dsr)
            else "DSR could not be computed."
        )
    )

    html = f"""
{_CSS}
<div class="ts-panel">
  <h2>&#128202; Tessera Evaluation Metrics</h2>

  <h3>Statistical Significance</h3>
  <table class="ts-table">
    <thead>
      <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
    </thead>
    <tbody>
      {metrics_rows}
    </tbody>
  </table>
  <p class="ts-note">
    <b>Interpretation:</b> {dsr_interp}
    PSR answers "is this SR genuine over this sample?";
    DSR additionally penalises for the {trial_count} configurations tried before picking this model.
    Bootstrap CI shows sampling uncertainty in the SR estimator.
    <span class="ts-warn">DSR &lt; 0.75 = discard.</span>
    <span class="ts-warn">DSR 0.75–0.95 = needs more data.</span>
    <span class="ts-pass">DSR ≥ 0.95 = deploy-ready.</span>
  </p>

  <h3>Stress-Window Performance</h3>
  <table class="ts-table">
    <thead>
      <tr>
        <th>Event</th><th>Window</th><th>IS/OOS</th><th>Data</th>
        <th>Total Return</th><th>Max DD</th><th>Annualised SR</th>
      </tr>
    </thead>
    <tbody>
      {stress_rows}
    </tbody>
  </table>
  <p class="ts-note">
    IS = in-sample (less meaningful).  OOS = out-of-sample (credibility evidence).
    "none" coverage means the strategy backtest window does not overlap this event.
    OOS losses are expected during tail events and are <em>not</em> a reason to reject
    the strategy — they confirm the evaluation is honest.
  </p>
</div>
<hr style="margin:0 auto;max-width:960px;border:none;border-top:1px solid #eee;">
"""
    return html
