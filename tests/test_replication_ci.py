"""Tests for CI utilities and ReplicationRunner."""


import pytest
import simpy

import simpy_stats
from simpy_stats.experiments.ci import _t_critical, ci_t, half_width_t
from simpy_stats.experiments.replication import ReplicationRunner
from simpy_stats.reporting.snapshot import Snapshot

# ---------------------------------------------------------------------------
# CI utilities
# ---------------------------------------------------------------------------

class TestCIT:
    def test_requires_min_two_values(self):
        with pytest.raises(ValueError):
            ci_t([1.0])

    def test_deterministic_values_tiny_hw(self):
        # When all values are identical, stdev=0 → half_width=0
        vals = [5.0] * 20
        ci = ci_t(vals)
        assert ci.mean == pytest.approx(5.0)
        assert ci.half_width == pytest.approx(0.0)

    def test_known_result(self):
        # Two-sample t-CI: n=2, mean=5, stdev=0 if both same
        vals = [4.0, 6.0]
        ci = ci_t(vals, alpha=0.05)
        assert ci.mean == pytest.approx(5.0)
        # stdev = sqrt(2), t(1, 0.05) = 12.706
        # hw = 12.706 * sqrt(2) / sqrt(2) = 12.706
        assert ci.half_width == pytest.approx(12.706, rel=0.01)

    def test_lower_upper_symmetric(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = ci_t(vals)
        assert ci.lower == pytest.approx(ci.mean - ci.half_width, rel=1e-9)
        assert ci.upper == pytest.approx(ci.mean + ci.half_width, rel=1e-9)

    def test_half_width_wrapper(self):
        vals = [1.0, 2.0, 3.0]
        assert half_width_t(vals) == pytest.approx(ci_t(vals).half_width)

    def test_t_critical_large_df_close_to_normal(self):
        # For large df, t critical approaches z=1.96
        t = _t_critical(1000, 0.05)
        assert abs(t - 1.960) < 0.05

    def test_t_critical_df1(self):
        t = _t_critical(1, 0.05)
        assert t == pytest.approx(12.706, rel=0.01)


# ---------------------------------------------------------------------------
# ReplicationRunner
# ---------------------------------------------------------------------------

def _make_rep_fn(base: float = 10.0, noise_scale: float = 0.5):
    """Factory for a deterministic rep_fn: returns a Snapshot with mean≈base."""
    import random

    def rep_fn(seed: int) -> Snapshot:
        rng = random.Random(seed)
        # Simulate a "run" by sampling a value
        value = base + rng.gauss(0, noise_scale)
        return Snapshot({"metric.mean": value, "count": float(rng.randint(50, 150))})

    return rep_fn


class TestReplicationRunner:
    def test_run_exact_n(self):
        runner = ReplicationRunner(_make_rep_fn())
        report = runner.run(n_reps=20)
        assert report.n_reps == 20
        assert len(report.snapshots) == 20

    def test_seeds_match(self):
        runner = ReplicationRunner(_make_rep_fn(base=5.0, noise_scale=0.0))
        report = runner.run(n_reps=10, seeds=list(range(10)))
        # With zero noise, all values == 5.0
        ms = report.metric_summary["metric.mean"]
        assert ms["mean"] == pytest.approx(5.0, abs=1e-9)
        assert ms["half_width"] == pytest.approx(0.0, abs=1e-9)

    def test_seed_length_mismatch_raises(self):
        runner = ReplicationRunner(_make_rep_fn())
        with pytest.raises(ValueError):
            runner.run(n_reps=5, seeds=[1, 2, 3])

    def test_metric_summary_keys(self):
        runner = ReplicationRunner(_make_rep_fn())
        report = runner.run(n_reps=10)
        ms = report.metric_summary
        assert "metric.mean" in ms
        for key in ("n", "mean", "stdev", "half_width", "lower", "upper"):
            assert key in ms["metric.mean"]

    def test_run_until_precision_finds_convergence(self):
        runner = ReplicationRunner(_make_rep_fn(base=100.0, noise_scale=1.0))
        report = runner.run_until_precision(
            metrics=["metric.mean"],
            rel_half_width=0.10,
            min_reps=5,
            max_reps=200,
        )
        ms = report.metric_summary["metric.mean"]
        hw_rel = ms["half_width"] / abs(ms["mean"])
        assert report.stop_reason == "precision reached"
        assert hw_rel <= 0.10

    def test_run_until_precision_max_reps(self):
        # Use huge noise so precision is never reached
        runner = ReplicationRunner(_make_rep_fn(base=1.0, noise_scale=1000.0))
        report = runner.run_until_precision(
            metrics=["metric.mean"],
            rel_half_width=0.001,
            min_reps=5,
            max_reps=15,
        )
        assert report.n_reps == 15
        assert report.stop_reason == "max reps reached"

    def test_stop_reason_attribute(self):
        runner = ReplicationRunner(_make_rep_fn())
        report = runner.run(5)
        assert report.stop_reason == "n_reps"

    def test_report_getitem(self):
        runner = ReplicationRunner(_make_rep_fn())
        report = runner.run(10)
        assert isinstance(report["metric.mean"], dict)


# ---------------------------------------------------------------------------
# Stats integration test
# ---------------------------------------------------------------------------

class TestStats:
    def test_full_workflow(self):
        env = simpy.Environment()
        stats = simpy_stats.Stats(env)
        wait = stats.tally("wait")
        arrivals = stats.counter("arrivals")
        queue = stats.level("queue")

        # Simulate events
        wait.observe(2.0)
        wait.observe(4.0)
        arrivals.inc(2)

        env.run(until=5)
        queue.update(3.0)
        env.run(until=10)

        snap = stats.finalize()

        assert "wait.mean" in snap
        assert snap["wait.mean"] == pytest.approx(3.0)
        assert snap["wait.n"] == 2.0
        assert snap["arrivals.count"] == 2.0
        assert "queue.time_mean" in snap
        # queue: 0 for [0,5), 3 for [5,10) → mean = 1.5
        assert snap["queue.time_mean"] == pytest.approx(1.5)

    def test_scope_repr(self):
        stats = simpy_stats.Stats()
        stats.tally("x")
        assert "Stats" in repr(stats)


T_VALUES = {
    (1, 0.05): 12.7062047, (4, 0.05): 2.7764451, (9, 0.05): 2.2621572,
    (29, 0.05): 2.0452296, (120, 0.05): 1.9799304, (150, 0.05): 1.9759053,
    (1000, 0.05): 1.9623391, (9, 0.01): 3.2498355, (4, 0.10): 2.1318468,
    (2, 0.001): 31.5990546, (30, 0.5): 0.6827557,
}


@pytest.mark.parametrize(("df", "alpha"), list(T_VALUES))
def test_t_critical_matches_the_exact_quantile(df, alpha):
    assert _t_critical(df, alpha) == pytest.approx(T_VALUES[(df, alpha)], abs=2e-6)


def test_t_critical_takes_an_alpha_that_is_not_a_round_float():
    assert _t_critical(9, 1 - 0.95) == pytest.approx(2.2621572, abs=2e-6)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_alpha_outside_zero_and_one_is_refused(alpha):
    with pytest.raises(ValueError):
        ci_t([1.0, 2.0, 3.0], alpha)


def test_ci_t_on_a_worked_sample():
    # mean 3, stdev sqrt(2.5), n 5: half width 2.7764451 * sqrt(2.5 / 5)
    result = ci_t([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.mean == pytest.approx(3.0)
    assert result.half_width == pytest.approx(1.9632432, abs=1e-6)


def _fixed_snapshots(values):
    values = iter(values)
    return lambda seed: Snapshot({"wait.mean": next(values), "served.count": 1.0})


def test_a_replication_without_observations_is_left_out_of_the_summary():
    nan = float("nan")
    runner = ReplicationRunner(_fixed_snapshots([4.0, nan, 6.0, nan, 5.0]))
    report = runner.run(5)
    assert report.metric_summary["wait.mean"]["n"] == 3
    assert report.metric_summary["wait.mean"]["mean"] == pytest.approx(5.0)
    assert report.metric_summary["served.count"]["n"] == 5


def test_precision_is_not_reached_on_a_metric_that_has_no_values():
    nan = float("nan")
    runner = ReplicationRunner(_fixed_snapshots([nan] * 8))
    report = runner.run_until_precision(["wait.mean"], min_reps=3, max_reps=8)
    assert report.stop_reason == "max reps reached"
    assert report.n_reps == 8


def test_the_report_uses_the_alpha_of_the_stopping_rule():
    values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.0]
    runner = ReplicationRunner(_fixed_snapshots(values), alpha=0.05)
    report = runner.run_until_precision(
        ["wait.mean"], alpha=0.10, rel_half_width=0.5, min_reps=10, max_reps=10
    )
    assert report.metric_summary["wait.mean"]["half_width"] == pytest.approx(
        half_width_t(values, 0.10)
    )


def test_a_metric_seen_in_one_replication_has_no_spread():
    report = ReplicationRunner(_fixed_snapshots([7.0])).run(1)
    summary = report.metric_summary["wait.mean"]
    assert summary["mean"] == 7.0
    assert summary["stdev"] != summary["stdev"]  # NaN
