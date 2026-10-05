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
        """Incorporate a new observation *x* into the running statistics.

        Raises :exc:`ValueError` for NaN or infinity: one such value would
        turn the mean and the variance into NaN for good.
        """
        if not math.isfinite(x):
            raise ValueError(f"observation must be a finite number, got {x!r}")
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
        """Running mean.  NaN when n == 0: no observation, no mean."""
        if self._n == 0:
            return math.nan
        return self._mean

    @property
    def m2(self) -> float:
        """Sum of squared deviations from the mean."""
        return self._m2

    @property
    def var_sample(self) -> float:
        """Sample variance (denominator n-1).  NaN when n < 2."""
        if self._n < 2:
            return math.nan
        return self._m2 / (self._n - 1)

    @property
    def var_pop(self) -> float:
        """Population variance (denominator n).  NaN when n == 0."""
        if self._n == 0:
            return math.nan
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
        return f"Welford(n={self._n}, mean={self.mean:.6g}, stdev={self.stdev_sample:.6g})"
