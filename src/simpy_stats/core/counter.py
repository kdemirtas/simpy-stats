"""Counter: event-count statistic with optional rate computation."""

from __future__ import annotations


class Counter:
    """Accumulates event counts over a simulation run.

    Parameters
    ----------
    name:
        Human-readable label used in snapshot keys.
    """

    __slots__ = ("name", "_count")

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._count: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def inc(self, k: int = 1) -> None:
        """Increment the counter by *k* (default 1)."""
        if k < 0:
            raise ValueError(f"Counter.inc requires k >= 0, got {k}")
        self._count += k

    def reset(self) -> None:
        """Reset count to zero."""
        self._count = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Current accumulated count."""
        return self._count

    def rate(self, duration: float) -> float:
        """Return count / duration.

        Parameters
        ----------
        duration:
            Elapsed simulation time.  Must be > 0.
        """
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        return self._count / duration

    def snapshot(self) -> dict[str, float]:
        """Return a flat dict of metrics with keys prefixed by *name*."""
        return {f"{self.name}.count": float(self._count)}

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Counter(name={self.name!r}, count={self._count})"
