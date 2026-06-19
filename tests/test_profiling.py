"""Tests for the latency instrumentation registry."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "server"))

import profiling


def setup_function(_):
    profiling.reset()


def test_timer_records_sample():
    with profiling.Timer("stage_a"):
        time.sleep(0.01)
    snap = profiling.snapshot()
    assert "stage_a" in snap
    assert snap["stage_a"]["count"] == 1
    assert snap["stage_a"]["avg_ms"] >= 9  # ~10ms sleep


def test_record_and_aggregate():
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        profiling.record("s", v)
    s = profiling.snapshot()["s"]
    assert s["count"] == 5
    assert s["avg_ms"] == 30.0
    assert s["max_ms"] == 50.0
    assert s["last_ms"] == 50.0
    assert s["p50_ms"] == 30.0


def test_percentile_p95():
    for v in range(1, 101):  # 1..100
        profiling.record("p", float(v))
    p = profiling.snapshot()["p"]
    # p95 of 1..100 is ~95.05
    assert 94 <= p["p95_ms"] <= 96


def test_profiled_decorator():
    @profiling.profiled("decorated")
    def work():
        return 42
    assert work() == 42
    assert profiling.snapshot()["decorated"]["count"] == 1


def test_reset_clears():
    profiling.record("x", 1.0)
    profiling.reset()
    assert profiling.snapshot() == {}


def test_bounded_window():
    # The rolling window caps stored samples; count reflects the cap, not total.
    for i in range(profiling._MAX_SAMPLES + 50):
        profiling.record("bounded", float(i))
    assert profiling.snapshot()["bounded"]["count"] == profiling._MAX_SAMPLES
