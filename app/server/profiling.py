"""Lightweight latency instrumentation.

A process-local registry of timing samples per named stage, plus a context
manager / decorators to record them. Exposed at GET /api/metrics so the full
pipeline can be profiled in production without external tooling.

Deliberately dependency-free and thread-safe; overhead is a dict lookup + append
per sample, negligible next to ML/LLM/network calls.
"""
from __future__ import annotations

import functools
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque

# Keep a bounded rolling window of recent samples per stage (ms).
_MAX_SAMPLES = 500
_samples: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=_MAX_SAMPLES))
_lock = Lock()


def record(stage: str, elapsed_ms: float) -> None:
    with _lock:
        _samples[stage].append(elapsed_ms)


class Timer:
    """Context manager: `with Timer("stage"): ...` records wall-clock ms."""

    def __init__(self, stage: str):
        self.stage = stage
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        record(self.stage, self.elapsed_ms)


def profiled(stage: str):
    """Decorator for sync functions."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with Timer(stage):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def profiled_async(stage: str):
    """Decorator for async functions."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                record(stage, (time.perf_counter() - t0) * 1000.0)
        return wrapper
    return deco


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def snapshot() -> dict:
    """Aggregate stats per stage: count, avg, p50, p95, max, last (all ms)."""
    out = {}
    with _lock:
        items = {k: list(v) for k, v in _samples.items()}
    for stage, vals in items.items():
        if not vals:
            continue
        s = sorted(vals)
        out[stage] = {
            "count": len(vals),
            "avg_ms": round(sum(vals) / len(vals), 2),
            "p50_ms": round(_percentile(s, 0.50), 2),
            "p95_ms": round(_percentile(s, 0.95), 2),
            "max_ms": round(s[-1], 2),
            "last_ms": round(vals[-1], 2),
        }
    return out


def reset() -> None:
    with _lock:
        _samples.clear()
