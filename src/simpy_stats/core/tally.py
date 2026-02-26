"""Tally: observation-based streaming statistics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .welford import Welford

if TYPE_CHECKING:
    pass


class Tally:
    """Collect point-in-time observations and compute streaming summary stats.

    Uses Welford's algorithm for numerically stable mean/variance.

    Parameters
    ----------
    name:
        Human-readable label used in snapshot keys.
    """

    __slots__ = ("name", "_welford", "_min", "_max")

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._welford: Welford = Welford()
        self._min: float = math.inf
        self._max: float = -math.inf

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def observe(self, x: float, t: float | None = None) -> None:  # noqa: ARG002
        """Record a new observation *x*.

        Parameters
        ----------
        x:
            The observed value.
        t:
            Optional simulation timestamp (stored for potential future use;
            not currently used in streaming statistics).
        """
        self._welford.update(x)
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    def reset(self) -> None:
        """Clear all observations."""
        self._welford = Welford()
        self._min = math.inf
        self._max = -math.inf

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of observations."""
        return self._welford.n

    @property
    def mean(self) -> float:
        """Sample mean.  Returns 0.0 when n == 0."""
        return self._welford.mean

    @property
    def var(self) -> float:
        """Sample variance (n-1 denominator).  Returns 0.0 when n < 2."""
        return self._welford.var_sample

    @property
    def stdev(self) -> float:
        """Sample standard deviation."""
        return self._welford.stdev_sample

    @property
    def min(self) -> float:
        """Minimum observed value.  Returns inf when n == 0."""
        return self._min

    @property
    def max(self) -> float:
        """Maximum observed value.  Returns -inf when n == 0."""
        return self._max

    def snapshot(self) -> dict[str, float]:
        """Return a flat dict of metrics with keys prefixed by *name*."""
        n = self.n
        return {
            f"{self.name}.n": float(n),
            f"{self.name}.mean": self.mean,
            f"{self.name}.stdev": self.stdev,
            f"{self.name}.min": self.min if n > 0 else float("nan"),
            f"{self.name}.max": self.max if n > 0 else float("nan"),
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Tally(name={self.name!r}, n={self.n}, "
            f"mean={self.mean:.6g}, stdev={self.stdev:.6g})"
        )
