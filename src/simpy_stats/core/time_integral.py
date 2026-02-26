"""Time-weighted integral accumulator for piecewise-constant signals."""

from __future__ import annotations


class TimeIntegral:
    """Accumulate the time-weighted area under a piecewise-constant step function.

    Typical use: track queue length or WIP over continuous simulation time.

    Parameters
    ----------
    initial:
        Initial value of the signal at *start_time*.
    start_time:
        Simulation clock when observation begins.
    """

    __slots__ = ("_start_time", "_last_t", "_last_value", "_area")

    def __init__(self, initial: float = 0.0, start_time: float = 0.0) -> None:
        self._start_time: float = start_time
        self._last_t: float = start_time
        self._last_value: float = initial
        self._area: float = 0.0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, new_value: float, t: float) -> None:
        """Record a level change to *new_value* occurring at time *t*.

        Parameters
        ----------
        new_value:
            The value the signal takes *from* time *t* onward.
        t:
            Current simulation time.  Must be >= last update time.
        """
        if t < self._last_t:
            raise ValueError(
                f"Time must be non-decreasing: got {t} after {self._last_t}"
            )
        self._area += self._last_value * (t - self._last_t)
        self._last_t = t
        self._last_value = new_value

    def close(self, t_end: float) -> None:
        """Close the final segment up to *t_end* without changing the value."""
        self.update(self._last_value, t_end)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def area(self) -> float:
        """Accumulated time-weighted area (integral), excluding the open segment."""
        return self._area

    @property
    def last_value(self) -> float:
        """Most recently recorded signal value."""
        return self._last_value

    @property
    def last_t(self) -> float:
        """Simulation time of the last recorded change."""
        return self._last_t

    def time_mean(self, t_end: float | None = None) -> float:
        """Time-average of the signal over [start_time, t_end].

        Parameters
        ----------
        t_end:
            End of the observation window.  If None, only the already-closed
            segments are included (i.e., up to the last recorded change).
        """
        if t_end is None:
            total_area = self._area
            elapsed = self._last_t - self._start_time
        else:
            if t_end < self._last_t:
                raise ValueError(
                    f"t_end={t_end} is before last recorded time {self._last_t}"
                )
            # include the open trailing segment from last_t to t_end
            total_area = self._area + self._last_value * (t_end - self._last_t)
            elapsed = t_end - self._start_time

        if elapsed == 0.0:
            return self._last_value
        return total_area / elapsed

    def __repr__(self) -> str:
        return (
            f"TimeIntegral(start={self._start_time}, last_t={self._last_t}, "
            f"last_value={self._last_value}, area={self._area:.6g})"
        )
