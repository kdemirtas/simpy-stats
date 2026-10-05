"""attach_resource_monitors: convenience helper to wire Levels to a SimPy Resource."""

from __future__ import annotations

import simpy

from ..core.level import Level


def attach_resource_monitors(
    resource: simpy.Resource,
    stats,
    prefix: str = "server",
) -> dict[str, Level]:
    """Attach queue-length and in-service Level monitors to a plain SimPy Resource.

    This is the non-subclassing alternative.  It monkey-patches the resource's
    ``_trigger_put`` / ``_trigger_get`` / ``release`` methods so that Level statistics are
    updated on every state change.

    Parameters
    ----------
    resource:
        A :class:`simpy.Resource` instance (not yet started / partially used).
    stats:
        A :class:`~simpy_stats.Stats` (or any object with a
        ``level(name)`` factory method).
    prefix:
        Metric name prefix (default ``"server"``).

    Returns
    -------
    dict
        ``{"queue_len": Level, "in_service": Level}``
    """
    env = resource._env
    queue_lvl: Level = stats.level(f"{prefix}.queue_len", initial=0)
    service_lvl: Level = stats.level(f"{prefix}.in_service", initial=0)

    _orig_trigger_put = resource._trigger_put  # type: ignore[attr-defined]
    _orig_trigger_get = resource._trigger_get  # type: ignore[attr-defined]
    _orig_release = resource.release

    def _sync() -> None:
        queue_lvl.update(len(resource.queue), float(env.now))
        service_lvl.update(resource.count, float(env.now))

    # SimPy removes a granted request from the queue only after ``_do_put``
    # returns, so the queue is read once the whole trigger pass is over.
    def _patched_trigger_put(get_event):
        _orig_trigger_put(get_event)
        _sync()

    def _patched_trigger_get(put_event):
        _orig_trigger_get(put_event)
        _sync()

    def _patched_release(request):
        result = _orig_release(request)
        _sync()
        return result

    resource._trigger_put = _patched_trigger_put  # type: ignore[method-assign]
    resource._trigger_get = _patched_trigger_get  # type: ignore[method-assign]
    resource.release = _patched_release  # type: ignore[method-assign]

    return {"queue_len": queue_lvl, "in_service": service_lvl}
