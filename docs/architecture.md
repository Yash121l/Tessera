# Architecture

TODO: System architecture diagram and component interaction overview.

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
