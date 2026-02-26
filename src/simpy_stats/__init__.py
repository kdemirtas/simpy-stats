"""simpy_stats — Arena-like statistics layer for SimPy.

Quick start::

    import simpy
    import simpy_stats

    env = simpy.Environment()
    stats = simpy_stats.Stats(env)

    wait = stats.tally("wait")
    arrivals = stats.counter("arrivals")
    queue_len = stats.level("queue_len")

    # ... run simulation ...

    snap = stats.finalize()
    print(simpy_stats.summary_table(snap))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core.counter import Counter
from .core.tally import Tally
from .core.level import Level
from .core.welford import Welford
from .reporting.snapshot import Snapshot
from .reporting.summary import summary_table
from .reporting.export import to_csv, to_json
from .experiments.ci import ci_t, half_width_t
from .experiments.replication import ReplicationRunner, ReplicationReport

if TYPE_CHECKING:
    import simpy

__all__ = [
    # Core stats
    "Counter",
    "Tally",
    "Level",
    "Welford",
    # Scope
    "Stats",
    # Reporting
    "Snapshot",
    "summary_table",
    "to_csv",
    "to_json",
    # Experiments
    "ci_t",
    "half_width_t",
    "ReplicationRunner",
    "ReplicationReport",
]

__version__ = "0.1.0"


class Stats:
    """Factory and registry for all statistics in a single simulation run.

    Parameters
    ----------
    env:
        SimPy environment (used as the default time source for Level stats).
    """

    def __init__(self, env: "simpy.Environment | None" = None) -> None:
        self._env = env
        self._tallies: dict[str, Tally] = {}
        self._counters: dict[str, Counter] = {}
        self._levels: dict[str, Level] = {}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def tally(self, name: str) -> Tally:
        """Return (or create) a :class:`Tally` registered under *name*."""
        if name not in self._tallies:
            self._tallies[name] = Tally(name)
        return self._tallies[name]

    def counter(self, name: str) -> Counter:
        """Return (or create) a :class:`Counter` registered under *name*."""
        if name not in self._counters:
            self._counters[name] = Counter(name)
        return self._counters[name]

    def level(self, name: str, initial: float = 0.0) -> Level:
        """Return (or create) a :class:`Level` registered under *name*.

        Parameters
        ----------
        name:
            Metric name.
        initial:
            Starting value of the level (used only on first creation).
        """
        if name not in self._levels:
            self._levels[name] = Level(name, env=self._env, initial=initial)
        return self._levels[name]

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self, t_end: float | None = None) -> Snapshot:
        """Close all Level segments and return a :class:`Snapshot`.

        Parameters
        ----------
        t_end:
            End time of the simulation run.  Defaults to ``env.now`` when an
            environment is attached, otherwise the last recorded level time.
        """
        if t_end is None and self._env is not None:
            t_end = float(self._env.now)

        metrics: dict[str, float] = {}

        for lvl in self._levels.values():
            lvl.finalize(t_end)
            metrics.update(lvl.snapshot())

        for tally in self._tallies.values():
            metrics.update(tally.snapshot())

        for counter in self._counters.values():
            metrics.update(counter.snapshot())

        return Snapshot(metrics)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def tallies(self) -> dict[str, Tally]:
        return self._tallies

    @property
    def counters(self) -> dict[str, Counter]:
        return self._counters

    @property
    def levels(self) -> dict[str, Level]:
        return self._levels

    def __repr__(self) -> str:
        return (
            f"Stats("
            f"tallies={list(self._tallies)}, "
            f"counters={list(self._counters)}, "
            f"levels={list(self._levels)})"
        )
