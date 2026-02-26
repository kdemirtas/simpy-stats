"""MonitoredStore: SimPy Store wrapper with automatic Level tracking."""

from __future__ import annotations

import simpy


class MonitoredStore(simpy.Store):
    """A :class:`simpy.Store` that tracks item count as a time-weighted Level.

    Statistics updated
    ------------------
    * ``<prefix>.items`` — number of items currently in the store (time-weighted).

    Parameters
    ----------
    env:
        SimPy environment.
    capacity:
        Maximum number of items (default unlimited).
    stats:
        A :class:`~simpy_stats.StatScope`.
    prefix:
        Prefix for metric names (default ``"store"``).
    """

    def __init__(
        self,
        env: simpy.Environment,
        capacity: float = float("inf"),
        stats=None,
        prefix: str = "store",
    ) -> None:
        super().__init__(env, capacity)
        self._items_level = None
        if stats is not None:
            self._items_level = stats.level(f"{prefix}.items", initial=0)

    def _do_put(self, event):  # type: ignore[override]
        result = super()._do_put(event)
        self._sync()
        return result

    def _do_get(self, event):  # type: ignore[override]
        result = super()._do_get(event)
        self._sync()
        return result

    def _sync(self) -> None:
        if self._items_level is not None:
            self._items_level.update(len(self.items), float(self._env.now))
