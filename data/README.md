# Data Directory

This directory is gitignored. It contains raw and processed market data.

## Structure

```
data/
├── raw/          # Raw OHLCV + trades from exchanges
├── processed/    # Cleaned, resampled bars
├── features/     # Computed feature matrices
└── labels/       # Generated labels (triple-barrier, etc.)
```

## Reproducing

Run `tessera ingest` with the appropriate config to populate this directory.
