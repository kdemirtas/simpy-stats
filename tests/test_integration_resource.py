"""Tests for MonitoredResource and attach_resource_monitors."""

import pytest
import simpy

import simpy_stats
from simpy_stats.simpy_integration.monitored_resource import MonitoredResource
from simpy_stats.simpy_integration.hooks import attach_resource_monitors


# ---------------------------------------------------------------------------
# MonitoredResource
# ---------------------------------------------------------------------------

def _simple_server_sim(env, stats, capacity=1):
    """One customer arrives at t=0, served until t=5."""
    resource = MonitoredResource(env, capacity=capacity, stats=stats, prefix="server")

    def customer():
        with resource.request() as req:
            yield req
            yield env.timeout(5)

    env.process(customer())
    return resource


class TestMonitoredResource:
    def test_queue_and_service_levels_created(self):
        env = simpy.Environment()
        stats = simpy_stats.StatScope(env)
        _simple_server_sim(env, stats)
        env.run()
        assert "server.queue_len" in stats.levels
        assert "server.in_service" in stats.levels

    def test_in_service_time_mean(self):
        env = simpy.Environment()
        stats = simpy_stats.StatScope(env)
        _simple_server_sim(env, stats)
        env.run(until=10)
        snap = stats.finalize()
        # server busy for t=[0,5), idle for [5,10): time_mean = 0.5
        assert snap["server.in_service.time_mean"] == pytest.approx(0.5)

    def test_no_stats_attached(self):
        """MonitoredResource works fine without stats (no crash)."""
        env = simpy.Environment()
        resource = MonitoredResource(env, capacity=1)
        assert resource._queue_level is None

        def customer():
            with resource.request() as req:
                yield req
                yield env.timeout(3)

        env.process(customer())
        env.run()  # should not raise


# ---------------------------------------------------------------------------
# attach_resource_monitors
# ---------------------------------------------------------------------------

class TestAttachResourceMonitors:
    def test_basic_attach(self):
        env = simpy.Environment()
        stats = simpy_stats.StatScope(env)
        resource = simpy.Resource(env, capacity=1)
        monitors = attach_resource_monitors(resource, stats, prefix="srv")

        assert "queue_len" in monitors
        assert "in_service" in monitors

    def test_utilization_tracked(self):
        env = simpy.Environment()
        stats = simpy_stats.StatScope(env)
        resource = simpy.Resource(env, capacity=1)
        attach_resource_monitors(resource, stats, prefix="srv")

        def customer():
            with resource.request() as req:
                yield req
                yield env.timeout(4)

        env.process(customer())
        env.run(until=10)
        snap = stats.finalize()
        # server busy for [0,4): in_service time_mean = 4/10 = 0.4
        assert snap["srv.in_service.time_mean"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Snapshot export tests
# ---------------------------------------------------------------------------

class TestExport:
    def test_to_csv(self):
        from simpy_stats.reporting.export import to_csv
        snap = simpy_stats.Snapshot({"a.mean": 1.0, "b.count": 5.0})
        csv_text = to_csv(snap)
        assert "a.mean" in csv_text
        assert "1.0" in csv_text

    def test_to_json(self):
        import json
        from simpy_stats.reporting.export import to_json
        snap = simpy_stats.Snapshot({"x": 3.14})
        data = json.loads(to_json(snap))
        assert data["x"] == pytest.approx(3.14)

    def test_to_csv_multiple(self):
        from simpy_stats.reporting.export import to_csv
        snaps = [
            simpy_stats.Snapshot({"a": 1.0}),
            simpy_stats.Snapshot({"a": 2.0}),
        ]
        csv_text = to_csv(snaps)
        lines = csv_text.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows

    def test_summary_table(self):
        snap = simpy_stats.Snapshot({"wait.mean": 2.5, "arrivals.count": 100.0})
        table = simpy_stats.summary_table(snap)
        assert "wait.mean" in table
        assert "2.5" in table
