"""M/M/1 queue example using simpy_stats.

Demonstrates:
- Tally for customer waiting times
- Level for time-average queue length
- ReplicationRunner with half-width stopping rule

Usage::

    uv run python examples/mm1_queue.py
"""

from __future__ import annotations

import random

import simpy

import simpy_stats
from simpy_stats import ReplicationRunner, StatScope, summary_table


# ---------------------------------------------------------------------------
# M/M/1 simulation
# ---------------------------------------------------------------------------

ARRIVAL_RATE = 0.8   # arrivals per time unit (λ)
SERVICE_RATE = 1.0   # services per time unit (μ)
SIM_TIME = 1_000.0   # warm-up + run length


def mm1_rep(seed: int) -> simpy_stats.Snapshot:
    """Run one M/M/1 replication and return a Snapshot."""
    rng = random.Random(seed)
    env = simpy.Environment()
    stats = StatScope(env)

    wait_time = stats.tally("wait_time")
    queue_len = stats.level("queue_len", initial=0)
    in_service = stats.level("in_service", initial=0)
    num_served = stats.counter("served")

    server = simpy.Resource(env, capacity=1)

    def customer():
        arrival = env.now
        queue_len.update(len(server.queue) + 1)  # joined queue
        with server.request() as req:
            yield req
            queue_len.update(len(server.queue))   # left queue / now in service
            in_service.update(server.count)
            wait_time.observe(env.now - arrival)
            service_time = rng.expovariate(SERVICE_RATE)
            yield env.timeout(service_time)
            in_service.update(server.count - 1)
            num_served.inc()

    def arrivals():
        while True:
            iat = rng.expovariate(ARRIVAL_RATE)
            yield env.timeout(iat)
            env.process(customer())

    env.process(arrivals())
    env.run(until=SIM_TIME)
    return stats.finalize()


# ---------------------------------------------------------------------------
# Run replications until 5% relative half-width on wait_time.mean
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner = ReplicationRunner(mm1_rep, alpha=0.05)
    report = runner.run_until_precision(
        metrics=["wait_time.mean", "queue_len.time_mean"],
        rel_half_width=0.05,
        min_reps=10,
        max_reps=100,
    )

    print(summary_table(report, title="M/M/1 Queue — Replication Report"))
    print(f"\nReplications run : {report.n_reps}")
    print(f"Stop reason      : {report.stop_reason}")

    # Theoretical M/M/1 values (ρ = λ/μ = 0.8)
    rho = ARRIVAL_RATE / SERVICE_RATE
    E_W = rho / (SERVICE_RATE * (1 - rho))      # mean waiting time in queue
    E_Lq = ARRIVAL_RATE * E_W                    # mean queue length (Little's law)
    print(f"\nTheoretical E[W] (wait in queue) : {E_W:.4f}")
    print(f"Theoretical E[Lq] (queue length) : {E_Lq:.4f}")
