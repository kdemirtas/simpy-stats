"""Confidence-interval utilities (t-based, standard library only)."""

from __future__ import annotations

import math
import statistics
from typing import NamedTuple


class CIResult(NamedTuple):
    mean: float
    lower: float
    upper: float
    half_width: float


# ---------------------------------------------------------------------------
# Student t quantile, standard library only
# ---------------------------------------------------------------------------


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b), by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _incomplete_beta(b, a, 1.0 - x)
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    fraction = d
    for m in range(1, 500):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            fraction *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return math.exp(log_front) * fraction / a


def _t_two_tail(t: float, df: int) -> float:
    """P(|T| > t) for Student's t with *df* degrees of freedom, t >= 0."""
    return _incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def _t_critical(df: int, alpha: float = 0.05) -> float:
    """Return the two-tailed t critical value for *df* degrees of freedom.

    The value t with P(|T| > t) = alpha, found by bisection on the exact
    distribution.  No table and no scipy: the same number on every machine,
    for any ``0 < alpha < 1`` and any ``df >= 1``.
    """
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            f"alpha is the significance level and must be between 0 and 1, got {alpha}"
            " (0.05 gives a 95 % interval)"
        )
    low, high = 0.0, 1.0
    while _t_two_tail(high, df) > alpha:
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _t_two_tail(middle, df) > alpha:
            low = middle
        else:
            high = middle
        if high - low <= 1e-13 * high:
            break
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ci_t(values: list[float], alpha: float = 0.05) -> CIResult:
    """Compute a two-sided *t*-based confidence interval.

    Parameters
    ----------
    values:
        Sample of independent replication outputs for a single metric.
    alpha:
        Significance level, not the confidence level: 0.05 gives a 95 % CI
        (the default), 0.10 a 90 % CI.  Must be between 0 and 1.

    Returns
    -------
    CIResult
        Named tuple with fields ``mean``, ``lower``, ``upper``, ``half_width``.

    Raises
    ------
    ValueError
        If fewer than 2 values are supplied, or alpha is not between 0 and 1.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"ci_t requires at least 2 values, got {n}")

    mu = statistics.mean(values)
    s = statistics.stdev(values)
    t_crit = _t_critical(n - 1, alpha)
    hw = t_crit * s / math.sqrt(n)
    return CIResult(mean=mu, lower=mu - hw, upper=mu + hw, half_width=hw)


def half_width_t(values: list[float], alpha: float = 0.05) -> float:
    """Return only the half-width of the *t*-based CI.

    Convenience wrapper around :func:`ci_t`.
    """
    return ci_t(values, alpha).half_width
