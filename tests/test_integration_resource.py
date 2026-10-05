"""Tests for MonitoredResource and attach_resource_monitors."""

import pytest
import simpy

import simpy_stats
from simpy_stats.simpy_integration.hooks import attach_resource_monitors
from simpy_stats.simpy_integration.monitored_resource import MonitoredResource

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
        stats = simpy_stats.Stats(env)
        _simple_server_sim(env, stats)
        env.run()
        assert "server.queue_len" in stats.levels
        assert "server.in_service" in stats.levels

    def test_in_service_time_mean(self):
        env = simpy.Environment()
        stats = simpy_stats.Stats(env)
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
        stats = simpy_stats.Stats(env)
        resource = simpy.Resource(env, capacity=1)
        monitors = attach_resource_monitors(resource, stats, prefix="srv")

        assert "queue_len" in monitors
        assert "in_service" in monitors

    def test_utilization_tracked(self):
        env = simpy.Environment()
        stats = simpy_stats.Stats(env)
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


def test_monitored_classes_are_exported_at_the_top_level():
    import simpy_stats
    from simpy_stats import simpy_integration

    for name in (
        "MonitoredResource",
        "MonitoredStore",
        "MonitoredContainer",
        "attach_resource_monitors",
    ):
        assert getattr(simpy_stats, name) is getattr(simpy_integration, name)
        assert name in simpy_stats.__all__


PLACED_AT = {1: 0, 2: 2, 3: 3, 4: 14, 5: 15}
SERVICE_TIME = {1: 4, 2: 3, 3: 5, 4: 2, 5: 3}


def _five_customers(env, server):
    """One server, five customers who wait 0, 2, 4, 0 and 1: the run ends at 19."""

    def customer(number):
        yield env.timeout(PLACED_AT[number])
        with server.request() as turn:
            yield turn
            yield env.timeout(SERVICE_TIME[number])

    for number in PLACED_AT:
        env.process(customer(number))
    env.run()


def test_monitored_resource_queue_length_counts_only_waiting_requests():
    env = simpy.Environment()
    stats = simpy_stats.Stats(env)
    server = simpy_stats.MonitoredResource(env, capacity=1, stats=stats, prefix="server")
    _five_customers(env, server)
    snap = stats.finalize()
    assert snap["server.queue_len.time_mean"] == pytest.approx(7 / 19)
    assert snap["server.in_service.time_mean"] == pytest.approx(17 / 19)


def test_attached_monitors_queue_length_counts_only_waiting_requests():
    env = simpy.Environment()
    stats = simpy_stats.Stats(env)
    server = simpy.Resource(env, capacity=1)
    simpy_stats.attach_resource_monitors(server, stats, prefix="server")
    _five_customers(env, server)
    snap = stats.finalize()
    assert snap["server.queue_len.time_mean"] == pytest.approx(7 / 19)
    assert snap["server.in_service.time_mean"] == pytest.approx(17 / 19)


def _one_holder_one_quitter(env, server, leave):
    """A holder keeps the server for 10; a second customer waits 2 and gives up."""

    def holder():
        with server.request() as turn:
            yield turn
            yield env.timeout(10)

    def quitter():
        if leave == "with":
            with server.request() as turn:
                yield turn | env.timeout(2)
        else:
            turn = server.request()
            yield turn | env.timeout(2)
            turn.cancel()

    env.process(holder())
    env.process(quitter())
    env.run()


@pytest.mark.parametrize("leave", ["with", "cancel"])
@pytest.mark.parametrize("attached", [False, True])
def test_a_request_that_gives_up_leaves_the_queue_level(leave, attached):
    env = simpy.Environment()
    stats = simpy_stats.Stats(env)
    if attached:
        server = simpy.Resource(env, capacity=1)
        simpy_stats.attach_resource_monitors(server, stats, prefix="server")
    else:
        server = simpy_stats.MonitoredResource(env, capacity=1, stats=stats, prefix="server")
    _one_holder_one_quitter(env, server, leave)
    snap = stats.finalize()
    assert snap["server.queue_len.time_mean"] == pytest.approx(2 / 10)
    assert snap["server.in_service.time_mean"] == pytest.approx(1.0)


def test_attached_monitors_follow_a_priority_resource():
    env = simpy.Environment()
    stats = simpy_stats.Stats(env)
    server = simpy.PriorityResource(env, capacity=1)
    simpy_stats.attach_resource_monitors(server, stats, prefix="server")
    served = []

    def customer(name, arrives, priority):
        yield env.timeout(arrives)
        with server.request(priority=priority) as turn:
            yield turn
            served.append(name)
            yield env.timeout(4)

    env.process(customer("first", 0, 0))
    env.process(customer("low", 1, 5))
    env.process(customer("high", 2, 1))
    env.run()
    snap = stats.finalize()
    assert served == ["first", "high", "low"]
    # low waits from 1 to 8, high from 2 to 4: 9 customer-minutes over 12 minutes
    assert snap["server.queue_len.time_mean"] == pytest.approx(9 / 12)
    assert snap["server.in_service.time_mean"] == pytest.approx(1.0)


def test_attached_monitors_follow_a_preemptive_resource():
    env = simpy.Environment()
    stats = simpy_stats.Stats(env)
    server = simpy.PreemptiveResource(env, capacity=1)
    simpy_stats.attach_resource_monitors(server, stats, prefix="server")

    def low():
        with server.request(priority=5) as turn:
            yield turn
            try:
                yield env.timeout(10)
            except simpy.Interrupt:
                pass

    def high():
        yield env.timeout(3)
        with server.request(priority=1) as turn:
            yield turn
            yield env.timeout(4)

    env.process(low())
    env.process(high())
    env.run()
    snap = stats.finalize()
    # the abandoned timeout of the preempted user keeps the run going to 10;
    # the server is in use from 0 to 7
    assert env.now == 10
    assert snap["server.queue_len.time_mean"] == pytest.approx(0.0)
    assert snap["server.in_service.time_mean"] == pytest.approx(7 / 10)


@pytest.mark.parametrize("attached", [False, True])
def test_a_request_left_open_at_the_end_of_the_run_is_cancelled_quietly(attached):
    env = simpy.Environment()
    stats = simpy_stats.Stats(env)
    if attached:
        server = simpy.Resource(env, capacity=1)
        simpy_stats.attach_resource_monitors(server, stats, prefix="server")
    else:
        server = simpy_stats.MonitoredResource(env, capacity=1, stats=stats, prefix="server")
    def customer():
        with server.request() as turn:
            yield turn
            yield env.timeout(10)

    processes = [env.process(customer()), env.process(customer())]
    env.run(until=5)
    snap = stats.finalize()
    # what Python does to a process that is still alive when the model is
    # dropped: it closes the generator, which leaves the ``with`` block
    for process in processes:
        process._generator.close()
    assert snap["server.queue_len.time_mean"] == pytest.approx(1.0)
    assert stats.finalize()["server.queue_len.time_mean"] == pytest.approx(1.0)
