"""Level: time-persistent (time-weighted) statistic for piecewise-constant signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .time_integral import TimeIntegral

if TYPE_CHECKING:
    import simpy


class Level:
    """Track the time-weighted mean of a piecewise-constant level (e.g. queue length).

    Parameters
    ----------
    name:
        Human-readable label used in snapshot keys.
    env:
        SimPy environment.  When provided, ``env.now`` is used as the default
        timestamp in :meth:`update`.  Pass ``None`` to supply timestamps
        manually.
    initial:
        Initial level value at the start of observation.
    start_time:
        Simulation time when observation begins.  Defaults to ``env.now`` when
        *env* is supplied, otherwise 0.0.
    """

    __slots__ = ("name", "_env", "_integral", "_finalized")

    def __init__(
        self,
        name: str,
        env: "simpy.Environment | None" = None,
        initial: float = 0.0,
        start_time: float | None = None,
    ) -> None:
        self.name: str = name
        self._env = env
        self._finalized: bool = False

        if start_time is None:
            start_time = float(env.now) if env is not None else 0.0

        self._integral: TimeIntegral = TimeIntegral(
            initial=initial, start_time=start_time
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, value: float, t: float | None = None) -> None:
        """Record that the level changed to *value* at time *t*.

        Parameters
        ----------
        value:
            New level value, effective from time *t* onward.
        t:
            Simulation time of the change.  Defaults to ``env.now`` when an
            environment was supplied, otherwise raises :exc:`ValueError`.
        """
        if self._finalized:
            raise RuntimeError("Level has been finalized; no further updates allowed.")
        t = self._resolve_t(t)
        self._integral.update(value, t)

    def finalize(self, t_end: float | None = None) -> None:
        """Close the final segment and mark the Level as finalized.

        Parameters
        ----------
        t_end:
            End time of the observation window.  Defaults to ``env.now``.
        """
        if self._finalized:
            return
        t_end = self._resolve_t(t_end)
        self._integral.close(t_end)
        self._finalized = True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current(self) -> float:
        """The current (most recent) level value."""
        return self._integral.last_value

    def mean(self, t_end: float | None = None) -> float:
        """Time-weighted mean of the level over the observation window.

        Parameters
        ----------
        t_end:
            End of the window.  Required when the Level has not been finalized
            and no env is attached.  When the Level *has* been finalized,
            ``t_end`` is ignored and the already-closed window is used.
        """
        if self._finalized:
            # Use only closed segments
            return self._integral.time_mean(t_end=None)
        if t_end is None:
            t_end = self._resolve_t(None)
        return self._integral.time_mean(t_end=t_end)

    def snapshot(self) -> dict[str, float]:
        """Return a flat dict of metrics with keys prefixed by *name*."""
        t_end = None if self._finalized else self._resolve_t(None)
        return {f"{self.name}.time_mean": self.mean(t_end=t_end if not self._finalized else None)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_t(self, t: float | None) -> float:
        if t is not None:
            return float(t)
        if self._env is not None:
            return float(self._env.now)
        raise ValueError(
            "No simulation time available: pass 't' explicitly or attach an env."
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Level(name={self.name!r}, current={self.current}, "
            f"finalized={self._finalized})"
        )
