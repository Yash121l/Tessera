# Operations Runbook

TODO: Deployment procedures, monitoring alerts, incident response.

## Kill Switch

Set `TESSERA_KILL_SWITCH=true` or update the config to immediately halt all trading.

## Common Issues

| Symptom | Check | Fix |
|---------|-------|-----|
| No fills | Exchange connectivity | Check API keys, rate limits |
| High latency | Prometheus dashboard | Restart bot, check network |
| Drawdown breach | Risk limits | Kill switch auto-triggers |
