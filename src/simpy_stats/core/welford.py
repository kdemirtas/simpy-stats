"""Welford one-pass streaming mean and variance accumulator."""

from __future__ import annotations

import math


class Welford:
    """Online mean/variance using Welford's algorithm (numerically stable).

    Reference: Welford (1962), Knuth TAOCP Vol. 2.
    """

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, x: float) -> None:
        """Incorporate a new observation *x* into the running statistics."""
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of observations."""
        return self._n

    @property
    def mean(self) -> float:
        """Running mean.  Returns 0.0 when n == 0."""
        return self._mean

    @property
    def m2(self) -> float:
        """Sum of squared deviations from the mean."""
        return self._m2

    @property
    def var_sample(self) -> float:
        """Sample variance (denominator n-1).  Returns 0.0 when n < 2."""
        if self._n < 2:
            return 0.0
        return self._m2 / (self._n - 1)

    @property
    def var_pop(self) -> float:
        """Population variance (denominator n).  Returns 0.0 when n == 0."""
        if self._n == 0:
            return 0.0
        return self._m2 / self._n

    @property
    def stdev_sample(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.var_sample)

    @property
    def stdev_pop(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.var_pop)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Welford(n={self._n}, mean={self._mean:.6g}, stdev={self.stdev_sample:.6g})"
