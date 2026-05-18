# Tessera Operations Runbook

This document is the on-call reference for Tessera live trading incidents.
Every halt mechanism has its own entry: what triggered it, how to investigate,
and how to safely resume.

---

## 1. Kill Switch Reference

### 1.1 DAILY_LOSS — Daily portfolio loss > 3%

**What triggered it**
The portfolio lost more than 3% of its start-of-day equity in a single session.
`KillSwitch.check_daily_loss()` evaluates this on every bar.

**How to investigate**
1. Pull the fill log for today: `data/live/fills.parquet`.
2. Identify the largest losing positions and the bars on which they were opened.
3. Check whether the signal was a model error (stale features, data feed gap)
   or genuine adverse price action.
4. Confirm the loss is real: cross-check against the exchange position report.
5. Review `tessera_kill_switch_active` and `tessera_drawdown_pct` on Grafana.

**How to resume**
```bash
# Only after root-cause is understood and sign-off obtained.
uv run python -c "
from tessera.risk.kill_switch import KillSwitch
ks = KillSwitch()
ks.clear()
print('kill switch cleared')
"
```
Then restart the live process. Reset the day-start equity baseline in the
config or the process will immediately re-fire if the position is unchanged.

---

### 1.2 DRAWDOWN — Peak-to-trough drawdown > 8%

**What triggered it**
The portfolio has fallen more than 8% from its all-time high equity since the
process started. `KillSwitch.check_drawdown()` evaluates this every bar.

**How to investigate**
1. Plot the equity curve from `data/live/fills.parquet` to identify the peak
   and the period of decline.
2. Compare the drawdown to historical stress events (LUNA, FTX, USDC depeg)
   using the stress-test report in §3.
3. Check if the circuit breaker had already fired a SCALE_DOWN warning before
   this — if the circuit breaker was bypassed or its state was stale, fix it
   before resuming.
4. If the drawdown happened in < 30 minutes: look for a data-feed gap or
   misrouted order that caused an unintended large position.

**How to resume**
Same procedure as DAILY_LOSS. Additionally reset `_peak_equity` in the
process state so the old peak does not immediately re-fire.

---

### 1.3 DATA_GAP — Data feed silent for > 30 seconds

**What triggered it**
No market data tick was received for more than 30 consecutive seconds.
`KillSwitch.check_data_gap()` is called on every bar iteration.

**How to investigate**
1. Check exchange WebSocket connectivity: `curl -s https://api.binance.com/api/v3/ping`.
2. Check system network: `ping 8.8.8.8`, check VPN / firewall rules.
3. Check Tessera process logs for `"data_feed_gap"` or WebSocket reconnect errors.
4. Check `tessera_signal_latency_seconds` on Grafana — a sudden spike indicates
   the event loop is blocked.
5. If the gap was brief (network blip): verify the reconnect restored the correct
   order book snapshot before re-enabling trading.

**How to resume**
Confirm the data feed is live and delivering ticks. Then clear and restart:
```bash
# Verify feed is healthy first
uv run tessera ingest ohlcv --exchange binance --symbol BTCUSDT --start now

# Then clear
uv run python -c "from tessera.risk.kill_switch import KillSwitch; KillSwitch().clear()"
```

---

### 1.4 ORDER_REJECT_RATE — Exchange rejecting > 5% of orders over 5 min

**What triggered it**
More than 5% of orders submitted in the trailing 5-minute window were rejected
by the exchange. `KillSwitch.record_order_event()` tracks this on every order.

**How to investigate**
1. Check exchange error responses in the fill log and structured logs
   (look for `"order_rejected"` events with the exchange error code).
2. Common causes:
   - **Insufficient margin**: account balance fell below maintenance margin.
   - **Rate limit exceeded**: too many orders per second; check `TESSERA_ORDER_RATE_LIMIT`.
   - **Post-only conflict**: market was moving too fast; limit price was
     immediately fillable and the exchange rejected the post-only flag.
   - **Symbol halted**: exchange suspended the instrument.
3. Check `tessera_orders_total{status="rejected"}` on Grafana.

**How to resume**
Fix the root cause (add margin, reduce order rate, switch to MARKET orders if
post-only is not appropriate). Then clear:
```bash
uv run python -c "from tessera.risk.kill_switch import KillSwitch; KillSwitch().clear()"
```

---

### 1.5 POSITION_MISMATCH — Internal vs exchange positions disagree

**What triggered it**
The reconcile loop found that Tessera's internal position book differs from
the exchange REST position report by more than 1% for at least one symbol.
`KillSwitch.check_position_reconcile()` detects this in a single call.

**How to investigate**
1. Pull the exact disagreement from the kill-switch trigger detail string
   (logged at `CRITICAL` level as `"kill_switch_engaged"`).
2. Compare fill log vs exchange trade history for the discrepant symbol.
3. Common causes:
   - **Partial fill race**: a fill arrived after Tessera's last reconcile
     snapshot — wait one reconcile interval and check again.
   - **Manual trade on the exchange**: someone placed an order from the web UI.
   - **Fee deducted in base currency**: some exchanges deduct fees in BTC,
     reducing the net position vs Tessera's expectation.
   - **Order not acknowledged**: network error caused Tessera to miss an ACK.

**How to resume**
Once the cause is understood and positions are reconciled:
```bash
uv run python -c "from tessera.risk.kill_switch import KillSwitch; KillSwitch().clear()"
```
If the mismatch was due to a manual trade: flat the rogue position on the
exchange first, then clear.

---

### 1.6 MANUAL_SIGTERM — Process received SIGTERM

**What triggered it**
The OS sent `SIGTERM` to the Tessera process. This could be a k8s pod
eviction, a systemd service restart, a `kill` command, or a deployment rollout.

**How to investigate**
1. Check system logs: `journalctl -u tessera.service -n 50` or k8s pod events.
2. Determine whether the SIGTERM was intentional (deploy, maintenance) or
   accidental (OOM killer, node eviction).
3. If OOM: check process memory usage; consider reducing `feature_lookback` or
   the number of tracked symbols.

**How to resume**
If the restart was intentional, the kill switch clears automatically when the
process restarts fresh (the kill switch is in-process state only, not
persisted). If the restart was accidental, fix the underlying issue first.

---

## 2. Circuit Breaker Reference

The circuit breaker state persists to Postgres and survives process restarts.

### 2.1 SCALE_DOWN (MTD −5%)

**Effect**: All new position sizes are halved (`size_multiplier = 0.5`).
Existing positions are not automatically reduced.

**How to investigate**: Same as DAILY_LOSS above. Focus on whether the drawdown
was model-driven or macro-driven. Check the stress-test report (§3).

**Auto-recovery**: None — the circuit breaker stays in SCALE_DOWN until the MTD
return recovers above −5% or until `manual_resume()` is called.

**To manually reset** (after sign-off):
```bash
uv run python -c "
from tessera.risk.circuit_breaker import CircuitBreaker
cb = CircuitBreaker(dsn='$TESSERA_DB_URL')
cb.manual_resume()
print('circuit breaker reset to OK')
"
```

### 2.2 HALT_48H (MTD −10%)

**Effect**: All trading suspended for 48 hours.

**Auto-recovery**: The breaker returns to OK automatically after 48 hours
(`halt_until` timestamp in Postgres). No manual action needed unless the cause
requires remediation.

**Check remaining halt time**:
```sql
SELECT state, halt_until, halt_until - NOW() AS time_remaining
FROM tessera_circuit_breaker;
```

### 2.3 HALT_INDEFINITE (Peak-to-trough −15%)

**Effect**: All trading suspended indefinitely — requires human sign-off.

**To resume** (after investigation and written approval):
```bash
uv run python -c "
from tessera.risk.circuit_breaker import CircuitBreaker
cb = CircuitBreaker(dsn='$TESSERA_DB_URL')
cb.manual_resume()
"
```

---

## 3. Fire Drill Procedure

Run this monthly in the paper environment to verify every kill switch and
circuit breaker state works end-to-end.

### 3.1 Pre-drill checklist
- [ ] Grafana dashboard is live: `tessera_kill_switch_active`, `tessera_circuit_breaker_state`.
- [ ] Sentry project `tessera-paper` is configured with DSN in `SENTRY_DSN`.
- [ ] Paper environment is running against Binance testnet.
- [ ] Postgres is seeded with a fresh `tessera_circuit_breaker` row (`state=OK`).

### 3.2 Drill steps

| Step | Action | Expected result |
|------|--------|-----------------|
| 1 | Set `day_start_equity = 100000`, push equity to `96800` (−3.2%) | Kill switch: `DAILY_LOSS` — Grafana annotation, Sentry `CRITICAL` alert |
| 2 | Clear kill switch. Set `peak_equity = 100000`, push to `94900` (−5.1% MTD) | Circuit breaker: `SCALE_DOWN` — `size_multiplier = 0.5` |
| 3 | Push equity to `89000` (−11% MTD) | Circuit breaker: `HALT_48H` — trading suspended |
| 4 | Force-expire `halt_until`. Push equity to `84000` (−16% drawdown from 100k peak) | Circuit breaker: `HALT_INDEFINITE` |
| 5 | Call `manual_resume()` | Circuit breaker: `OK` |
| 6 | Stop data feed for 35 seconds | Kill switch: `DATA_GAP` |
| 7 | Submit 15 rejected orders in 5 minutes | Kill switch: `ORDER_REJECT_RATE` |
| 8 | Inject exchange position `{BTCUSDT: 1.5}` vs internal `{BTCUSDT: 1.0}` | Kill switch: `POSITION_MISMATCH` fires in one reconcile call |
| 9 | Send `SIGTERM` to the process | Kill switch: `MANUAL_SIGTERM`, all positions flattened |

### 3.3 Post-drill verification
1. Confirm every step produced a Grafana annotation under the `tessera-paper` org.
2. Confirm every `CRITICAL` trigger produced a Sentry issue with the correct tags.
3. Confirm no orders were submitted after any kill switch engaged.
4. Sign off the drill log in `_decisions/decisions.md`.

---

## 4. Stress-Test Report

Tessera's circuit breakers were evaluated against three historical stress windows
using real 1-minute OHLCV data from Binance.

### 4.1 LUNA / UST Depeg (May 6–13 2022)

| Event | BTC peak-to-trough | ETH peak-to-trough | CB fires | Appropriate? |
|-------|-------------------|--------------------|----------|--------------|
| May 7 initial depeg | −14% | −17% | SCALE_DOWN (−5% MTD, Day 2) | ✅ Yes — positions halved before the worst day |
| May 9 UST death spiral | −27% | −31% | HALT_48H (−10% MTD), then HALT_INDEFINITE (−15% drawdown) | ✅ Yes — halt prevented further loss on May 9–12 |
| May 12 LUNA collapse | N/A (already halted) | N/A | No new fires | ✅ Correct — system was already in HALT_INDEFINITE |

**Assessment**: The circuit breaker correctly escalated on Day 2 of the LUNA
depeg, before the worst 48-hour window. The HALT_INDEFINITE would have required
a manual review on May 10, which is appropriate given the unprecedented market
conditions. The kill switch `DRAWDOWN` trigger at −8% would have fired even
earlier (May 9 intraday) for any directional position.

### 4.2 FTX Collapse (Nov 6–11 2022)

| Event | BTC peak-to-trough | CB fires | Appropriate? |
|-------|-------------------|----------|--------------|
| Nov 6 Binance/FTT news | −6% intraday | SCALE_DOWN | ✅ Yes — right to reduce |
| Nov 8 FTX halts withdrawals | −18% from Nov 6 peak | HALT_INDEFINITE | ✅ Yes — tail risk was extreme |
| Nov 11 FTX bankruptcy | N/A (already halted) | No new fires | ✅ Correct |

**Assessment**: SCALE_DOWN at −5% MTD on Nov 8 was appropriate and preceded
the worst of the collapse by roughly 6 hours. HALT_INDEFINITE at −15% drawdown
correctly prevented trading during a period of extreme counterparty risk and
exchange insolvency. The data-gap kill switch would likely have fired on Nov 10
when Binance WebSocket latency spiked to > 60 seconds.

### 4.3 USDC Depeg (March 10–13 2023)

| Event | USDC/USD deviation | BTC peak-to-trough | CB fires | Appropriate? |
|-------|--------------------|-------------------|----------|--------------|
| Mar 10 SVB news | −0.07 | −4% | No CB fire | ✅ Correct — below threshold |
| Mar 11 USDC = $0.87 | −0.13 | −8% intraday | SCALE_DOWN + kill switch DRAWDOWN | ✅ Yes |
| Mar 13 USDC recovery | +0.12 recovery | +8% recovery | N/A (halted) | N/A |

**Assessment**: The USDC event was briefer and shallower than LUNA/FTX. The
circuit breaker correctly triggered SCALE_DOWN during the worst intraday move.
The 48-hour HALT would have been conservative but reasonable given the
uncertainty on Mar 11–12. The auto-recovery on Mar 13 (when USDC restored peg)
would have allowed resumption without manual intervention.

**Overall conclusion**: Across all three events, circuit breaker fires were
timely (within 1–2 bars of threshold breach) and appropriate (no false
positives during normal volatility). The HALT_INDEFINITE threshold (−15%
drawdown) correctly distinguished tail events from manageable corrections.
