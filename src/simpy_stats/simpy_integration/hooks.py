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
    ``_do_put`` / ``_do_get`` / ``release`` methods so that Level statistics are
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

    _orig_do_put = resource._do_put  # type: ignore[attr-defined]
    _orig_do_get = resource._do_get  # type: ignore[attr-defined]
    _orig_release = resource.release

    def _patched_do_put(event):
        result = _orig_do_put(event)
        queue_lvl.update(len(resource.queue), float(env.now))
        service_lvl.update(resource.count, float(env.now))
        return result

    def _patched_do_get(event):
        result = _orig_do_get(event)
        queue_lvl.update(len(resource.queue), float(env.now))
        service_lvl.update(resource.count, float(env.now))
        return result

    def _patched_release(request):
        result = _orig_release(request)
        queue_lvl.update(len(resource.queue), float(env.now))
        service_lvl.update(resource.count, float(env.now))
        return result

    resource._do_put = _patched_do_put  # type: ignore[method-assign]
    resource._do_get = _patched_do_get  # type: ignore[method-assign]
    resource.release = _patched_release  # type: ignore[method-assign]

    return {"queue_len": queue_lvl, "in_service": service_lvl}
