# Tessera Demo

This directory contains screen-recording assets for demonstrating Tessera end-to-end.

## Quick start

```bash
# From repo root:
bash paper/demo/record_demo.sh
```

Run this script while recording in **OBS**, **Loom**, or **QuickTime**.
Use [demo_script.md](demo_script.md) as your voiceover script.

## Files

| File | Purpose |
|---|---|
| `record_demo.sh` | Automated demo runner (backtest → tear sheet → Grafana) |
| `demo_script.md` | Voiceover script with talking points per section |
| `cast.cast` | Asciinema recording of `make backtest` on fixture data |

## Loom recording

<!-- TODO: replace with Loom URL after recording -->
**Loom URL:** _not yet recorded — run `bash paper/demo/record_demo.sh` and upload_

## Asciinema replay

```bash
# Install asciinema if needed:
brew install asciinema      # macOS

# Replay the terminal recording:
asciinema play paper/demo/cast.cast
```

## What the demo shows

1. **Backtest** (`make backtest` or pytest smoke test) — deterministic fixture run
2. **Tear sheet** — Sharpe, DSR, max drawdown, cumulative PnL chart
3. **Prometheus** at `localhost:9090` — live metrics scrape
4. **Grafana** at `localhost:3000` — Tessera dashboard with PnL, risk, latency rows
