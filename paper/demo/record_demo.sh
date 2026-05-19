#!/usr/bin/env bash
# record_demo.sh — runs the full Tessera demo sequence for screen recording.
#
# Prerequisites: uv, docker (for Grafana/Prometheus), asciinema.
# Usage: ./paper/demo/record_demo.sh
#
# Run this script while recording with OBS / Loom / QuickTime.
# Voiceover script: paper/demo/demo_script.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

_open() {
  if command -v xdg-open &>/dev/null; then
    xdg-open "$1" &
  elif command -v open &>/dev/null; then
    open "$1" &
  else
    echo "Open manually: $1"
  fi
}

echo "=== Tessera Demo: Starting ==="
echo ""

# Step 1: Run backtest against fixture data (fast, deterministic)
echo "--- Step 1: Backtest smoke test against fixture data ---"
uv run pytest tests/integration/test_backtest_smoke.py -v --tb=short
echo ""

# Step 2: Generate tear sheet
echo "--- Step 2: Generating tear sheet ---"
LAST_RUN=$(ls -t data/backtest_runs/ 2>/dev/null | head -1 || echo "")
if [[ -n "$LAST_RUN" ]]; then
  uv run tessera report backtest --run-id "$LAST_RUN" --output paper/demo/ 2>/dev/null || \
    echo "(report subcommand stub — tear sheet in data/backtest_runs/$LAST_RUN/)"
fi
echo ""

# Step 3: Start observability stack
echo "--- Step 3: Starting Prometheus + Grafana ---"
if command -v docker &>/dev/null; then
  docker compose up -d prometheus grafana 2>/dev/null || \
    docker-compose up -d prometheus grafana 2>/dev/null || \
    echo "(docker compose not configured — skip)"
  sleep 3
else
  echo "(docker not available — skip)"
fi

# Step 4: Open relevant URLs in browser
echo ""
echo "--- Step 4: Opening URLs ---"
_open "http://localhost:9090"       # Prometheus
_open "http://localhost:3000"       # Grafana
echo ""

echo "=== Demo sequence complete. ==="
echo "If recording: stop your screen capture now."
echo "See paper/demo/demo_script.md for voiceover text."
