# simpy-stats

Arena-like statistics layer for [SimPy](https://simpy.readthedocs.io/): streaming `Counter`, `Tally`, and time-weighted `Level` statistics, replication management, and t-based confidence intervals — all with no mandatory dependencies beyond SimPy itself.

## Features

- **`Counter`** — event counts and optional rates
- **`Tally`** — observation-based streaming mean, variance, min, max (Welford algorithm)
- **`Level`** — time-weighted average of a piecewise-constant signal (queue length, WIP, utilization)
- **`StatScope`** — factory + registry: one call per name, `finalize()` → `Snapshot`
- **`ReplicationRunner`** — independent replications with half-width stopping rules
- **`ci_t`** — two-sided *t*-based confidence intervals (built-in table; scipy optional)
- **SimPy integrations** — `MonitoredResource`, `MonitoredStore`, `MonitoredContainer`, `attach_resource_monitors()`

## Installation

```bash
pip install simpy-stats
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add simpy-stats
```

## Quick start

```python
import simpy
import simpy_stats

env = simpy.Environment()
stats = simpy_stats.StatScope(env)

wait = stats.tally("wait_time")
arrivals = stats.counter("arrivals")
queue_len = stats.level("queue_len")

# ... run simulation ...

snap = stats.finalize()
print(simpy_stats.summary_table(snap))
```

### Replications with half-width stopping

```python
from simpy_stats import ReplicationRunner

def my_rep(seed: int) -> simpy_stats.Snapshot:
    env = simpy.Environment()
    stats = simpy_stats.StatScope(env)
    # ... build and run model ...
    return stats.finalize()

runner = ReplicationRunner(my_rep)
report = runner.run_until_precision(
    metrics=["wait_time.mean"],
    rel_half_width=0.05,   # 5% relative half-width
    min_reps=10,
    max_reps=200,
)
print(simpy_stats.summary_table(report))
```

### Automatic resource monitoring

```python
from simpy_stats.simpy_integration import MonitoredResource

server = MonitoredResource(env, capacity=1, stats=stats, prefix="server")
# server.queue_len and server.in_service Levels are updated automatically
```

## Development

```bash
uv sync --extra dev   # install deps
uv run pytest -q      # run tests
uv run ruff check .   # lint
```

## Milestones

| | |
|---|---|
| M1 | Core stats + StatScope + 70 tests |
| M2 | ReplicationRunner + CI / half-width |
| M3 | MonitoredResource + helpers + examples |
| M4 | Docs + PyPI release |

## License

MIT
