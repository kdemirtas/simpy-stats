"""Confidence-interval utilities (t-based with optional scipy fallback)."""

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
# t-critical-value table (two-tailed, alpha=0.05 and alpha=0.10)
# Covers df 1..120 + inf.  Generated once at module import.
# ---------------------------------------------------------------------------

# Sparse lookup for common cases; we interpolate for missing df.
_T_TABLE_05: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}
_T_TABLE_10: dict[int, float] = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
    6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
    11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
    16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
    25: 1.708, 30: 1.697, 40: 1.684, 60: 1.671, 120: 1.658,
}

_ALPHA_TO_TABLE: dict[float, dict[int, float]] = {
    0.05: _T_TABLE_05,
    0.10: _T_TABLE_10,
}

_NORMAL_APPROX: dict[float, float] = {0.05: 1.960, 0.10: 1.645}


def _t_critical(df: int, alpha: float = 0.05) -> float:
    """Return the two-tailed t critical value for *df* degrees of freedom.

    Uses scipy.stats.t.ppf when scipy is available; otherwise falls back to a
    built-in table with linear interpolation, then to the normal approximation
    for large df.
    """
    try:
        from scipy.stats import t as _t_dist  # type: ignore[import]

        return float(_t_dist.ppf(1.0 - alpha / 2.0, df))
    except ImportError:
        pass

    table = _ALPHA_TO_TABLE.get(alpha)
    if table is None:
        # Fall through to normal approximation
        z = _NORMAL_APPROX.get(alpha)
        if z is None:
            # Last resort: use 1.96 for alpha=0.05 vicinity
            z = -math.log(alpha / 2.0) ** 0.5 * 1.4  # rough approximation
        return z

    # Find bracketing keys
    keys = sorted(table.keys())
    if df <= keys[0]:
        return table[keys[0]]
    if df >= keys[-1]:
        # Use normal approximation for very large df
        return _NORMAL_APPROX.get(alpha, 1.960)

    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return table[lo]

    # Linear interpolation
    frac = (df - lo) / (hi - lo)
    return table[lo] + frac * (table[hi] - table[lo])


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
        Significance level (default 0.05 → 95 % CI).

    Returns
    -------
    CIResult
        Named tuple with fields ``mean``, ``lower``, ``upper``, ``half_width``.

    Raises
    ------
    ValueError
        If fewer than 2 values are supplied.
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
