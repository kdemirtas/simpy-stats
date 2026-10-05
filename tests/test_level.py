"""Tests for Level and TimeIntegral."""

import math

import pytest
import simpy

import simpy_stats
from simpy_stats.core.level import Level
from simpy_stats.core.time_integral import TimeIntegral

# ---------------------------------------------------------------------------
# TimeIntegral
# ---------------------------------------------------------------------------

class TestTimeIntegral:
    def test_constant_signal(self):
        ti = TimeIntegral(initial=3.0, start_time=0.0)
        # Signal stays at 3 from 0 to 10
        assert ti.time_mean(t_end=10.0) == pytest.approx(3.0)

    def test_step_at_midpoint(self):
        # Level=0 for [0,5), then level=10 for [5,10)
        # time_mean = (0*5 + 10*5) / 10 = 5.0
        ti = TimeIntegral(initial=0.0, start_time=0.0)
        ti.update(10.0, t=5.0)
        assert ti.time_mean(t_end=10.0) == pytest.approx(5.0)

    def test_multiple_steps(self):
        # [0,2): value=1  → area=2
        # [2,5): value=3  → area=9
        # [5,10): value=2 → area=10
        # total area=21, elapsed=10, mean=2.1
        ti = TimeIntegral(initial=1.0, start_time=0.0)
        ti.update(3.0, t=2.0)
        ti.update(2.0, t=5.0)
        assert ti.time_mean(t_end=10.0) == pytest.approx(2.1)

    def test_time_mean_at_zero_elapsed(self):
        ti = TimeIntegral(initial=5.0, start_time=0.0)
        # A window of no length has no time average
        assert math.isnan(ti.time_mean(t_end=0.0))

    def test_non_zero_start_time(self):
        # Level=2 from t=5 to t=15: elapsed=10, mean=2
        ti = TimeIntegral(initial=2.0, start_time=5.0)
        assert ti.time_mean(t_end=15.0) == pytest.approx(2.0)

    def test_close(self):
        ti = TimeIntegral(initial=4.0, start_time=0.0)
        ti.close(10.0)
        # After close, open trailing segment is gone (it was already added)
        assert ti.time_mean() == pytest.approx(4.0)

    def test_decreasing_time_raises(self):
        ti = TimeIntegral(initial=0.0, start_time=0.0)
        ti.update(1.0, t=5.0)
        with pytest.raises(ValueError):
            ti.update(2.0, t=3.0)

    def test_t_end_before_last_t_raises(self):
        ti = TimeIntegral(initial=0.0, start_time=0.0)
        ti.update(1.0, t=5.0)
        with pytest.raises(ValueError):
            ti.time_mean(t_end=3.0)


# ---------------------------------------------------------------------------
# Level
# ---------------------------------------------------------------------------

class TestLevel:
    def test_constant_level(self):
        lvl = Level("q", initial=0.0, start_time=0.0)
        assert lvl.mean(t_end=10.0) == pytest.approx(0.0)

    def test_step_change(self):
        # [0,4)=0 → area=0; [4,8)=5 → area=20; [8,10)=0 → area=0
        # total area=20, elapsed=10, time_mean=2.0
        lvl = Level("q", initial=0.0, start_time=0.0)
        lvl.update(5.0, t=4.0)
        lvl.update(0.0, t=8.0)
        assert lvl.mean(t_end=10.0) == pytest.approx(2.0)

    def test_step_change_correct(self):
        # [0,4): 0 → area=0
        # [4,8): 5 → area=20
        # [8,10): 0 → area=0
        # total=20, elapsed=10, mean=2.0
        lvl = Level("q", initial=0.0, start_time=0.0)
        lvl.update(5.0, t=4.0)
        lvl.update(0.0, t=8.0)
        assert lvl.mean(t_end=10.0) == pytest.approx(2.0)

    def test_finalize(self):
        lvl = Level("q", initial=3.0, start_time=0.0)
        lvl.finalize(t_end=6.0)
        assert lvl.mean() == pytest.approx(3.0)

    def test_finalize_idempotent(self):
        lvl = Level("q", initial=2.0, start_time=0.0)
        lvl.finalize(t_end=5.0)
        lvl.finalize(t_end=5.0)  # second call is a no-op
        assert lvl.mean() == pytest.approx(2.0)

    def test_update_after_finalize_raises(self):
        lvl = Level("q", initial=0.0, start_time=0.0)
        lvl.finalize(t_end=5.0)
        with pytest.raises(RuntimeError):
            lvl.update(1.0, t=6.0)

    def test_no_env_no_t_raises(self):
        lvl = Level("q", initial=0.0, start_time=0.0)
        with pytest.raises(ValueError):
            lvl.update(1.0)  # no env, no t

    def test_with_env(self):
        import simpy
        env = simpy.Environment()
        lvl = Level("q", env=env, initial=0.0)
        # advance time to 5, update level
        env.run(until=5)
        lvl.update(2.0)
        env.run(until=10)
        lvl.finalize()
        assert lvl.mean() == pytest.approx(1.0)  # 0*5 + 2*5 / 10 = 1.0

    def test_snapshot(self):
        lvl = Level("queue", initial=1.0, start_time=0.0)
        lvl.finalize(t_end=10.0)
        snap = lvl.snapshot()
        assert "queue.time_mean" in snap
        assert snap["queue.time_mean"] == pytest.approx(1.0)

    def test_current_property(self):
        lvl = Level("q", initial=3.0, start_time=0.0)
        assert lvl.current == 3.0
        lvl.update(7.0, t=5.0)
        assert lvl.current == 7.0

    def test_repr(self):
        lvl = Level("q", initial=0.0)
        assert "Level" in repr(lvl)


class TestStatsFinalize:
    def test_two_updates_at_the_same_time_add_no_area(self):
        ti = TimeIntegral(initial=0.0, start_time=0.0)
        ti.update(5.0, 2.0)
        ti.update(1.0, 2.0)
        assert ti.time_mean(t_end=4.0) == pytest.approx(2 / 4)

    def test_a_level_created_late_averages_over_its_own_window(self):
        env = simpy.Environment()
        stats = simpy_stats.Stats(env)
        early = stats.level("early", initial=1)
        env.run(until=10)
        late = stats.level("late", initial=1)
        early.update(0)
        late.update(0)
        env.run(until=20)
        snap = stats.finalize()
        assert snap["early.time_mean"] == pytest.approx(10 / 20)
        assert math.isnan(snap["late.time_mean"]) is False
        assert snap["late.time_mean"] == pytest.approx(0.0)

    def test_finalize_twice_at_the_same_time_gives_the_same_numbers(self):
        env = simpy.Environment()
        stats = simpy_stats.Stats(env)
        queue = stats.level("queue", initial=2)
        env.run(until=10)
        assert stats.finalize()["queue.time_mean"] == stats.finalize()["queue.time_mean"] == 2
        with pytest.raises(RuntimeError):
            queue.update(3)

    def test_finalize_again_at_a_later_time_is_refused(self):
        env = simpy.Environment()
        stats = simpy_stats.Stats(env)
        stats.level("queue", initial=2)
        env.run(until=10)
        stats.finalize()
        env.run(until=20)
        with pytest.raises(RuntimeError):
            stats.finalize()

    def test_finalize_without_env_ends_each_level_at_its_last_change(self):
        stats = simpy_stats.Stats()
        queue = stats.level("queue", initial=0)
        queue.update(4, t=5)
        queue.update(0, t=10)
        assert stats.finalize()["queue.time_mean"] == pytest.approx(20 / 10)
