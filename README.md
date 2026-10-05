# simpy-stats

Arena-like statistics layer for [SimPy](https://simpy.readthedocs.io/): streaming `Counter`, `Tally`, and time-weighted `Level` statistics, replication management, and t-based confidence intervals — all with no mandatory dependencies beyond SimPy itself.

## Features

- **`Counter`** — event counts and optional rates
- **`Tally`** — observation-based streaming mean, variance, min, max (Welford algorithm)
- **`Level`** — time-weighted average of a piecewise-constant signal (queue length, WIP, utilization)
- **`Stats`** — factory + registry: one call per name, `finalize()` → `Snapshot`
- **`ReplicationRunner`** — independent replications with half-width stopping rules
- **`ci_t`** — two-sided *t*-based confidence intervals, exact *t* quantile for any level, standard library only
- **SimPy integrations** — `MonitoredResource`, `MonitoredStore`, `MonitoredContainer`, `attach_resource_monitors()`

## Installation

Works with SimPy 4.1 (tested on 4.1.0, 4.1.1 and 4.1.2).

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
stats = simpy_stats.Stats(env)

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
    stats = simpy_stats.Stats(env)
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
from simpy_stats import MonitoredResource

server = MonitoredResource(env, capacity=1, stats=stats, prefix="server")
# server.queue_len and server.in_service Levels are updated automatically
```

## Changes

### 0.2.0

Fixes that change numbers:

- `MonitoredResource` and `attach_resource_monitors` counted the request that was being granted as waiting, so `queue_len` was too high. One server and five customers who wait 0, 2, 4, 0 and 1 over 19 time units gave 1.105; the right value is 7/19 = 0.368.
- A request cancelled by hand (`request.cancel()` outside a `with` block) stayed in `queue_len` until the next request or release.
- `ci_t` without scipy returned a wrong or complex half width for any `alpha` other than exactly 0.05 or 0.10, and used the normal value from 120 degrees of freedom on. The *t* quantile is now computed exactly for any `0 < alpha < 1`; scipy is no longer used, and the `scipy` extra is gone.
- An empty `Tally` reported mean 0.0 and stdev 0.0. It now reports NaN (also the stdev of a single observation), and `ReplicationRunner` leaves a replication without a value out of that metric's summary instead of averaging a zero in.
- `run_until_precision` stopped with "precision reached" on a metric that was NaN, and its report used the runner's `alpha` instead of the `alpha` of the stopping rule.
- The time average of a window of no length is NaN, not the last value.

Other changes:

- `MonitoredResource`, `MonitoredStore`, `MonitoredContainer` and `attach_resource_monitors` are importable from `simpy_stats`.
- `attach_resource_monitors` is tested on `PriorityResource` and `PreemptiveResource`.
- `Tally.observe` and `Welford.update` refuse NaN and infinity.
- `Stats.finalize()` ends the measurement. A second call with another end time raises `RuntimeError`. Without an environment and without `t_end`, each Level ends at its last recorded change.
- `alpha` is checked to be between 0 and 1 (it is the significance level: 0.05 gives a 95 % interval).
- SimPy is required as `>=4.1,<4.2` (it was `>=4.0`). The monitored classes extend SimPy's own resource classes through private methods, so the package is tied to the SimPy releases its tests ran on: 4.1.0, 4.1.1 and 4.1.2.

## Development

```bash
uv sync --extra dev   # install deps
uv run pytest -q      # run tests
uv run ruff check .   # lint
```

## Milestones

| | |
|---|---|
| M1 | Core stats + Stats + 70 tests |
| M2 | ReplicationRunner + CI / half-width |
| M3 | MonitoredResource + helpers + examples |
| M4 | Docs + PyPI release |

## Citation

If you use simpy-stats in academic work, please cite it:

```bibtex
@software{demirtas2026simpy_stats,
  author  = {Demirtas, Kerem},
  title   = {simpy-stats: Arena-like statistics layer for SimPy},
  year    = {2026},
  url     = {https://github.com/kdemirtas/simpy-stats},
  version = {0.1.0}
}
```

Or use the **"Cite this repository"** button on the GitHub page (powered by `CITATION.cff`).

## License

MIT
