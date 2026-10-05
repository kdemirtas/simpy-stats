"""ReplicationRunner: run multiple replications and aggregate CI statistics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from ..reporting.snapshot import Snapshot
from .ci import ci_t

# ---------------------------------------------------------------------------
# ReplicationReport
# ---------------------------------------------------------------------------

@dataclass
class MetricStats:
    """Per-metric summary across replications."""
    n: int
    mean: float
    stdev: float
    half_width: float
    lower: float
    upper: float
    values: list[float] = field(default_factory=list, repr=False)


@dataclass
class ReplicationReport:
    """Aggregated results from a :class:`ReplicationRunner` run.

    Attributes
    ----------
    snapshots:
        Raw :class:`~simpy_stats.reporting.snapshot.Snapshot` from each rep.
    metric_summary:
        ``{metric_key: MetricStats}`` across all replications.
    n_reps:
        Number of replications actually executed.
    stop_reason:
        ``"precision reached"`` or ``"max reps reached"``.
    """

    snapshots: list[Snapshot]
    metric_summary: dict[str, dict[str, float]]
    n_reps: int
    stop_reason: str = "n_reps"

    def __getitem__(self, metric: str) -> dict[str, float]:
        return self.metric_summary[metric]

    def keys(self):
        return self.metric_summary.keys()


# ---------------------------------------------------------------------------
# ReplicationRunner
# ---------------------------------------------------------------------------

RepFn = Callable[[int], Snapshot]  # seed -> Snapshot


class ReplicationRunner:
    """Run independent replications of a SimPy model and aggregate statistics.

    Parameters
    ----------
    rep_fn:
        Callable ``(seed: int) -> Snapshot``.  The function must create a
        fresh SimPy environment each call (seeds ensure independence).
    alpha:
        Default significance level for confidence intervals (default 0.05).
    """

    def __init__(self, rep_fn: RepFn, alpha: float = 0.05) -> None:
        self._rep_fn = rep_fn
        self._alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_reps: int,
        seeds: list[int] | None = None,
    ) -> ReplicationReport:
        """Run exactly *n_reps* replications.

        Parameters
        ----------
        n_reps:
            Number of replications to execute.
        seeds:
            Optional list of integer seeds (length must equal *n_reps*).
            If ``None``, seeds are drawn from ``range(n_reps)``.
        """
        if seeds is None:
            seeds = list(range(n_reps))
        if len(seeds) != n_reps:
            raise ValueError(
                f"len(seeds)={len(seeds)} must equal n_reps={n_reps}"
            )

        snapshots: list[Snapshot] = []
        for seed in seeds:
            snap = self._rep_fn(seed)
            snapshots.append(snap)

        return self._build_report(snapshots, stop_reason="n_reps")

    def run_until_precision(
        self,
        metrics: list[str],
        alpha: float | None = None,
        rel_half_width: float = 0.05,
        min_reps: int = 10,
        max_reps: int = 200,
        seed_start: int = 0,
    ) -> ReplicationReport:
        """Run replications until relative half-width criterion is met.

        The runner executes at least *min_reps* replications, then checks
        whether every metric in *metrics* satisfies::

            half_width / |mean| <= rel_half_width

        Additional replications are added one at a time until the criterion
        is met for all listed metrics or *max_reps* is reached.

        Parameters
        ----------
        metrics:
            Metric keys (dot notation) to monitor for the stopping rule.
        alpha:
            Significance level.  Defaults to the runner's ``alpha``.
        rel_half_width:
            Target relative half-width (fraction of |mean|).
        min_reps:
            Minimum number of replications before checking.
        max_reps:
            Hard cap on replications.
        seed_start:
            Seeds are assigned sequentially from this value.
        """
        alpha = alpha if alpha is not None else self._alpha
        snapshots: list[Snapshot] = []
        stop_reason = "max reps reached"

        for i in range(max_reps):
            seed = seed_start + i
            snap = self._rep_fn(seed)
            snapshots.append(snap)

            n = len(snapshots)
            if n < min_reps:
                continue

            # Check stopping criterion for all monitored metrics
            if self._precision_met(snapshots, metrics, alpha, rel_half_width):
                stop_reason = "precision reached"
                break

        return self._build_report(snapshots, stop_reason=stop_reason, alpha=alpha)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_values(
        snapshots: list[Snapshot], metric: str
    ) -> list[float]:
        # A replication that has no value for the metric (missing, or NaN
        # because nothing was observed) is left out; ``n`` reports how many
        # replications the summary rests on.
        return [s[metric] for s in snapshots if metric in s and math.isfinite(s[metric])]

    def _precision_met(
        self,
        snapshots: list[Snapshot],
        metrics: list[str],
        alpha: float,
        rel_half_width: float,
    ) -> bool:
        for m in metrics:
            vals = self._collect_values(snapshots, m)
            if len(vals) < 2:
                return False
            ci = ci_t(vals, alpha)
            if not (math.isfinite(ci.mean) and math.isfinite(ci.half_width)):
                return False
            mu = abs(ci.mean)
            if mu == 0.0:
                # If mean is zero, check absolute half-width instead
                if ci.half_width > rel_half_width:
                    return False
            else:
                if ci.half_width / mu > rel_half_width:
                    return False
        return True

    def _build_report(
        self, snapshots: list[Snapshot], stop_reason: str, alpha: float | None = None
    ) -> ReplicationReport:
        alpha = alpha if alpha is not None else self._alpha
        if not snapshots:
            return ReplicationReport(
                snapshots=[],
                metric_summary={},
                n_reps=0,
                stop_reason=stop_reason,
            )

        # Gather all metric keys that appear in at least one snapshot
        all_keys: list[str] = []
        seen: set[str] = set()
        for snap in snapshots:
            for k in snap.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        metric_summary: dict[str, dict[str, float]] = {}
        for k in all_keys:
            vals = self._collect_values(snapshots, k)
            if len(vals) < 2:
                # Single value — report without CI
                mu = vals[0] if vals else float("nan")
                metric_summary[k] = {
                    "n": float(len(vals)),
                    "mean": mu,
                    "stdev": float("nan"),
                    "half_width": float("nan"),
                    "lower": float("nan"),
                    "upper": float("nan"),
                }
            else:
                ci = ci_t(vals, alpha)
                import statistics as _stats
                metric_summary[k] = {
                    "n": float(len(vals)),
                    "mean": ci.mean,
                    "stdev": _stats.stdev(vals),
                    "half_width": ci.half_width,
                    "lower": ci.lower,
                    "upper": ci.upper,
                }

        return ReplicationReport(
            snapshots=snapshots,
            metric_summary=metric_summary,
            n_reps=len(snapshots),
            stop_reason=stop_reason,
        )
