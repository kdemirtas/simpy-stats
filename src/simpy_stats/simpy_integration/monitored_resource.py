"""MonitoredResource: SimPy Resource wrapper with automatic Level tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import simpy

from .hooks import sync_on_cancel

if TYPE_CHECKING:
    from ..core.level import Level


class MonitoredResource(simpy.Resource):
    """A :class:`simpy.Resource` that automatically updates Level statistics.

    Statistics updated
    ------------------
    * ``<prefix>.queue_len`` — number of requests waiting (time-weighted).
    * ``<prefix>.in_service`` — number of servers busy (time-weighted).

    Parameters
    ----------
    env:
        SimPy environment.
    capacity:
        Resource capacity (number of concurrent servers).
    stats:
        A :class:`~simpy_stats.Stats` (or any object with a
        ``level(name)`` factory method).
    prefix:
        Prefix for metric names (default ``"resource"``).
    """

    def __init__(
        self,
        env: simpy.Environment,
        capacity: int = 1,
        stats=None,
        prefix: str = "resource",
    ) -> None:
        super().__init__(env, capacity)
        self._prefix = prefix
        self._queue_level: Level | None = None
        self._service_level: Level | None = None

        if stats is not None:
            self._queue_level = stats.level(f"{prefix}.queue_len", initial=0)
            self._service_level = stats.level(f"{prefix}.in_service", initial=0)

    # ------------------------------------------------------------------
    # Override SimPy hooks
    # ------------------------------------------------------------------

    def _trigger_put(self, get_event):  # type: ignore[override]
        # SimPy removes a granted request from the queue only after ``_do_put``
        # returns, so the queue is read here, once the whole pass is over.
        super()._trigger_put(get_event)
        self._sync_levels()

    def _trigger_get(self, put_event):  # type: ignore[override]
        super()._trigger_get(put_event)
        self._sync_levels()

    def request(self, *args, **kwargs):
        request = super().request(*args, **kwargs)
        sync_on_cancel(request, self._sync_levels)
        return request

    def release(self, request):
        evt = super().release(request)
        self._sync_levels()
        return evt

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_levels(self) -> None:
        t = float(self._env.now)
        if self._queue_level is not None:
            self._queue_level.update(len(self.queue), t)
        if self._service_level is not None:
            self._service_level.update(self.count, t)
