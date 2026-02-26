"""Inventory (s, S) policy example using simpy_stats.

An (s, S) inventory model:
- Demand arrives as a Poisson process (rate λ).
- Each demand unit reduces on-hand inventory by 1.
- When inventory hits or falls below reorder point *s*, order up to *S*.
- Lead time is zero (instantaneous replenishment) for simplicity.

Metrics tracked:
- on_hand.time_mean   : time-average on-hand inventory (Level)
- fill_rate           : fraction of demand met immediately (Tally / Counter)
- lost_sales.count    : total lost sales (Counter)

Usage::

    uv run python examples/inventory_sS.py
"""

from __future__ import annotations

import random

import simpy

import simpy_stats
from simpy_stats import ReplicationRunner, Stats, summary_table


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

S = 20        # order-up-to level
s = 5         # reorder point
DEMAND_RATE = 2.0   # demands per time unit (Poisson)
SIM_TIME = 500.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def inventory_rep(seed: int) -> simpy_stats.Snapshot:
    rng = random.Random(seed)
    env = simpy.Environment()
    stats = Stats(env)

    on_hand = stats.level("on_hand", initial=float(S))
    lost_sales = stats.counter("lost_sales")
    demand_total = stats.counter("demand_total")
    fill_rate_tally = stats.tally("fill_rate_obs")

    inventory = S  # mutable state

    def demands():
        nonlocal inventory
        while True:
            iat = rng.expovariate(DEMAND_RATE)
            yield env.timeout(iat)

            demand_total.inc()
            if inventory > 0:
                inventory -= 1
                on_hand.update(float(inventory))
                fill_rate_tally.observe(1.0)
                # Replenish if below reorder point
                if inventory <= s:
                    inventory = S
                    on_hand.update(float(inventory))
            else:
                lost_sales.inc()
                fill_rate_tally.observe(0.0)

    env.process(demands())
    env.run(until=SIM_TIME)
    return stats.finalize()


# ---------------------------------------------------------------------------
# Run replications
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner = ReplicationRunner(inventory_rep, alpha=0.05)
    report = runner.run_until_precision(
        metrics=["on_hand.time_mean", "fill_rate_obs.mean"],
        rel_half_width=0.05,
        min_reps=10,
        max_reps=100,
    )

    print(summary_table(report, title=f"Inventory (s={s}, S={S}) — Replication Report"))
    print(f"\nReplications run : {report.n_reps}")
    print(f"Stop reason      : {report.stop_reason}")
