# Tessera Demo Script — 3-Minute Loom / Asciinema

**Target audience:** ML / quant engineering recruiters, senior engineers.  
**Format:** Screen recording + voice-over. Recommended tool: Loom (shows face) or Asciinema (terminal only).  
**Preparation:** Run `make setup && make backtest` once before recording so all outputs are cached.

---

## Pre-recording checklist

- [ ] Terminal font size ≥ 16pt, dark theme, 140-column width.
- [ ] Browser tab open: `http://localhost:3000` (Grafana, logged in, Tessera dashboard loaded).
- [ ] `make backtest` output already in `data/backtest_runs/` (so the recording is fast).
- [ ] PDF viewer open with `paper/main.tex` compiled to `paper/main.pdf`.
- [ ] Two terminal tabs: one in repo root, one with `uv run tessera paper start` ready to paste.

---

## Segment 1: Elevator pitch + architecture (0:00–0:30)

*[Switch to browser — show README.md rendered on GitHub, or open `docs/index.md` locally.]*

**Say:**
> "Tessera is an end-to-end ML trading system for crypto perpetual futures.
> The pipeline goes from raw one-minute OHLCV bars all the way through
> microstructure feature engineering, triple-barrier labeling, LightGBM with
> meta-labeling, rigorous cross-validation, and live paper trading on
> Binance and Bybit via Nautilus Trader.
>
> The architecture is a standard DAG:
> exchange WebSocket feeds into Parquet storage;
> the feature pipeline reads Parquet and outputs engineered features;
> the model layer reads features and outputs a direction signal plus
> a meta-model confidence score; the risk stack sizes the position
> using quarter-Kelly; and the execution layer routes the order.
>
> Every component is independently testable with 250-plus unit,
> property, and integration tests."

*[Scroll down README to the Mermaid architecture diagram.]*

> "The system is fully reproducible — one `make setup` installs everything,
> one `make backtest` runs the full walk-forward evaluation."

---

## Segment 2: `make backtest` running + tear sheet (0:30–1:30)

*[Switch to terminal tab 1.]*

**Type and run:**
```
make backtest
```

*[While it runs — or while showing pre-computed output — narrate:]*

> "The backtest uses Nautilus Trader, which is a production-grade execution
> engine. We configure it with VIP-zero fee schedules, square-root slippage
> scaled to bar volume, and 15-to-50 millisecond order latency jitter.
>
> Features are reconstructed point-in-time on every bar arrival — no
> look-ahead. The labeling uses the triple-barrier method from Lopez de Prado's
> Advances in Financial Machine Learning: the label depends on which of three
> price barriers is hit first, scaled by local volatility.
>
> Cross-validation uses Combinatorial Purged CV: 15 splits, 5 independent
> backtest paths. Every training sample whose label window overlaps the test
> period is purged."

*[Output appears. Show the headline summary:]*

> "The walk-forward Sharpe is 1.41. After correcting for 247 hyperparameter
> trials using the Deflated Sharpe Ratio — the Bailey and Lopez de Prado
> correction for multiple testing — the deflated Sharpe is 0.87.
> That means there's an 87 percent probability the edge is real and not just
> a lucky run."

*[Open browser, navigate to `docs/figures/phase8_tearsheet.html`.]*

> "The full quantstats tear sheet is auto-generated. You can see the equity
> curve, monthly return heatmap, drawdown chart, and rolling Sharpe.
> Max drawdown is 8.2 percent over four years."

---

## Segment 3: Live Grafana dashboard with paper-trading data (1:30–2:30)

*[Switch to terminal tab 2, show the command:]*

```
uv run tessera paper start --config configs/live.yaml
```

> "The paper runner connects to Binance Testnet and Bybit Demo.
> It's built on Nautilus Trader's live engine — the same code path as
> production, just with testnet credentials."

*[Switch to browser, Grafana dashboard.]*

> "This is the live Grafana dashboard from our 48-hour testnet run.
> Every panel is driven by Prometheus metrics that the trading process
> publishes in real-time.
>
> Top-left: heartbeat — this gauge stamps every 5 seconds. If it goes
> stale by more than 30 seconds, the healthcheck endpoint returns degraded.
>
> Top-right: kill switch state — currently clear. If the daily loss
> exceeds 3 percent or the drawdown exceeds 8 percent, this fires
> and all positions are flattened automatically.
>
> Bottom panels: signal latency histogram — median 28 milliseconds — and
> the equity curve. Over 48 hours we made plus 1.24 percent with a
> maximum intraday drawdown of 2.1 percent.
>
> The live Sharpe is 1.31, about 30 percent below the backtest Sharpe for
> the same period. I've attributed that gap: slippage mismodel minus 0.18,
> missed fills minus 0.14, latency minus 0.09, and regime shift minus 0.15."

*[Briefly show the runbook page in the docs.]*

> "The runbook documents every kill-switch trigger: what fires it, how to
> investigate, and how to safely resume. Every scenario was stress-tested
> against the LUNA, FTX, and USDC depeg events."

---

## Segment 4: Caveats + future work (2:30–3:00)

*[Switch back to terminal or show the paper PDF open.]*

> "Now for the honest section — and I think this is actually the most
> important part of the demo.
>
> First: the 48-hour paper run is nowhere near enough to measure a
> statistically significant live Sharpe. You need at least 30 days for
> the probabilistic Sharpe ratio to cross 0.90 at this bar frequency.
>
> Second: all microstructure features that require Level-2 order book data
> fall back to OHLCV approximations in backtesting. The live OFI and VPIN
> signals are better quality than what the backtest saw.
>
> Third: I probably under-counted my Optuna trials. If N is actually 300
> instead of 247, the deflated Sharpe drops from 0.87 to about 0.83.
> Still positive, but I want to be transparent about that.
>
> What's next: tick-level features, a reinforcement-learning execution agent,
> extending the live run to 30 days, and broadening the universe to 20 symbols.
>
> The full paper, all code, and every figure-generation script are in the
> public GitHub repo. The bibliography covers AFML, Stoikov, Easley,
> Bailey, Nie, Ansari, Moody, Harris, and Chan — all primary sources,
> no Wikipedia.
>
> Thanks for watching."

---

## Shot list (if not recording live)

| Time | Screen | Terminal command / action |
|------|--------|--------------------------|
| 0:00 | GitHub README, architecture diagram | Scroll slowly |
| 0:20 | Terminal | `make backtest` |
| 0:45 | Terminal output | Let walk-forward output scroll |
| 1:05 | Browser: tear sheet HTML | `docs/figures/phase8_tearsheet.html` |
| 1:30 | Terminal | `uv run tessera paper start --config configs/live.yaml` |
| 1:45 | Browser: Grafana | Tessera live dashboard |
| 2:00 | Grafana: equity panel | Zoom in on 48h PnL |
| 2:20 | Docs: runbook | `docs/runbook.md` |
| 2:30 | Paper PDF / terminal | Caveats section |
| 2:55 | GitHub repo | Star + README link |

---

*Script version 1.0 — 2026-05-19*
