"""MonitoredContainer: SimPy Container wrapper with automatic Level tracking."""

from __future__ import annotations

import simpy


class MonitoredContainer(simpy.Container):
    """A :class:`simpy.Container` that tracks level as a time-weighted stat.

    Statistics updated
    ------------------
    * ``<prefix>.level`` — container fill level (time-weighted).

    Parameters
    ----------
    env:
        SimPy environment.
    capacity:
        Container capacity (default 1).
    init:
        Initial fill level (default 0).
    stats:
        A :class:`~simpy_stats.StatScope`.
    prefix:
        Prefix for metric names (default ``"container"``).
    """

    def __init__(
        self,
        env: simpy.Environment,
        capacity: float = 1,
        init: float = 0,
        stats=None,
        prefix: str = "container",
    ) -> None:
        super().__init__(env, capacity, init)
        self._fill_level = None
        if stats is not None:
            self._fill_level = stats.level(f"{prefix}.level", initial=init)

    def _do_put(self, event):  # type: ignore[override]
        result = super()._do_put(event)
        self._sync()
        return result

    def _do_get(self, event):  # type: ignore[override]
        result = super()._do_get(event)
        self._sync()
        return result

    def _sync(self) -> None:
        if self._fill_level is not None:
            self._fill_level.update(self.level, float(self._env.now))
