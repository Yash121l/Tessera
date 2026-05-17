# Architecture

## Data Flow

```
Exchange WS → Ingest → Parquet/DuckDB → Features → Model → Signal → Execution → Exchange API
```

## Services

| Service | Purpose |
|---------|---------|
| bot | Main trading process |
| redis | Signal queue, rate limiting |
| postgres | Orders, fills, positions |
| prometheus | Metrics collection |
| grafana | Dashboards |

## Cross-cutting Concerns

Every module in Tessera depends on three foundational services that are initialized
once at process startup (via the CLI bootstrap) and flow through the entire system:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Bootstrap                                │
│  1. TesseraSettings ← .env + ENV vars + configs/*.yaml              │
│  2. configure_logging(level, json) → structlog                      │
│  3. seed_everything(seed) → numpy, torch, random                    │
│  4. start_metrics_server(port) → prometheus /metrics endpoint       │
└──────────┬──────────────────────┬────────────────────┬──────────────┘
           │                      │                    │
           ▼                      ▼                    ▼
┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
│   Config Layer   │  │   Logging Layer     │  │  Metrics Layer   │
│                  │  │                     │  │                  │
│ TesseraSettings  │  │ structlog + stdlib  │  │ prometheus_client│
│ DataConfig       │  │ JSON (prod) / TTY   │  │ Counters, Gauges │
│ FeatureConfig    │  │                     │  │ Histograms       │
│ ModelConfig      │  │ Fields:             │  │                  │
│ BacktestConfig   │  │ - timestamp (ISO)   │  │ Exposed at:      │
│ LiveConfig       │  │ - level             │  │ :9090/metrics    │
│ RiskConfig       │  │ - logger name       │  │                  │
│                  │  │ - event             │  │ Scraped by:      │
│ load_yaml(path)  │  │ - run_id (ctx)      │  │ Prometheus       │
│ seed_everything()│  │ - strategy (ctx)    │  │ → Grafana        │
│                  │  │ - symbol (ctx)      │  │                  │
└──────────────────┘  └─────────────────────┘  └──────────────────┘
           │                      │                    │
           └──────────────────────┼────────────────────┘
                                  │
                                  ▼
           ┌──────────────────────────────────────────┐
           │            Application Modules            │
           │  data/ features/ models/ strategies/     │
           │  backtest/ execution/ risk/ live/         │
           └──────────────────────────────────────────┘
```

### Config Layer

- **TesseraSettings**: Loads from environment variables (prefixed `TESSERA_`) and `.env` file.
  Secrets use `SecretStr` to prevent accidental logging.
- **YAML Configs**: Each `configs/*.yaml` file maps to a typed Pydantic model via `load_yaml()`.
  Models provide validation, defaults, and IDE autocomplete.
- **Reproducibility**: `seed_everything(seed)` deterministically seeds Python random, NumPy,
  and PyTorch. Every experiment is reproducible given the same seed + data version.

### Logging Layer

- **structlog** wraps Python's stdlib logging with structured key-value context.
- In **dev**: colored console output for readability.
- In **paper/live**: JSON lines for machine parsing (shipped to log aggregation).
- Context variables (`run_id`, `strategy`, `symbol`) are bound at the CLI level and
  automatically included in every downstream log line without explicit passing.

### Metrics Layer

- **prometheus_client** exposes an HTTP `/metrics` endpoint.
- Pre-defined metrics cover: order flow, fill rates, signal/order latency,
  position sizes, PnL, and drawdown.
- Grafana dashboards (in `infra/grafana/`) visualize these for real-time monitoring.
- The `tessera_drawdown_pct` gauge feeds the kill-switch logic in the risk module.
