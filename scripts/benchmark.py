"""Latency micro-benchmarks for Phase 5 optimizations.

Measures the optimizations we can run without a camera/GPU:
  1. PyTorch Arabic predictor: INT8 dynamic quantization ON vs OFF
     (both with torch.inference_mode), on synthetic landmark input.
  2. gloss→sentence cache: cold (LLM call) vs warm (cache hit), using a fake
     LLM with a fixed simulated latency.

Run:  python scripts/benchmark.py
Outputs a small table; the numbers behind docs/LATENCY.md come from here.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app", "server"))


def _time(fn, runs):
    # Warm up once, then time `runs` iterations; return mean ms.
    fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / runs


def bench_torch(runs=50):
    import numpy as np
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[torch bench skipped] {e}")
        return

    from models.sign_predictor import SignLanguagePredictor
    model_path = os.path.join(ROOT, "models", "best_model.pth")
    cmap = os.path.join(ROOT, "models", "class_mapping.json")
    if not os.path.exists(model_path):
        print("[torch bench skipped] best_model.pth not found")
        return

    frames = np.random.rand(60, 543, 3).astype("float32")

    results = {}
    for label, int8 in (("fp32 (inference_mode)", "0"), ("int8 dynamic", "1")):
        os.environ["TORCH_INT8"] = int8
        pred = SignLanguagePredictor(model_path=model_path, class_mapping_path=cmap, device="cpu")
        ms = _time(lambda: pred.predict_sign_from_landmarks(frames, 640, 480), runs)
        results[label] = ms
        print(f"  Arabic predictor [{label:24}] : {ms:7.2f} ms/inference")

    if "fp32 (inference_mode)" in results and "int8 dynamic" in results:
        base = results["fp32 (inference_mode)"]
        q = results["int8 dynamic"]
        print(f"  -> INT8 speedup: {base / q:.2f}x ({base:.2f} -> {q:.2f} ms)")


def bench_gloss_cache(runs=20):
    import gloss
    from providers.base import LLMProvider

    class _SlowLLM(LLMProvider):
        name = "slow"
        def generate(self, prompt, temperature=0.0):
            time.sleep(0.05)  # simulate a 50 ms LLM round-trip
            return "I am going to the store."

    gloss.clear_sentence_cache()
    tokens = ["STORE", "I", "GO"]

    # Cold path: bypass cache by passing the provider explicitly.
    cold = _time(lambda: gloss.gloss_to_sentence(tokens, "english", llm=_SlowLLM()), runs)

    # Warm the module cache once (llm=None routes through default provider; we
    # prime the cache entry directly to avoid a real network call).
    gloss._SENTENCE_CACHE[("STORE I GO", "english")] = "I am going to the store."
    warm = _time(lambda: gloss.gloss_to_sentence(tokens, "english"), runs)

    print(f"  gloss->sentence  [cold, 50ms LLM]        : {cold:7.2f} ms")
    print(f"  gloss->sentence  [warm, cache hit]       : {warm:7.3f} ms")
    print(f"  -> cache speedup: {cold / max(warm, 1e-6):.0f}x")


if __name__ == "__main__":
    print("=== Phase 5 latency micro-benchmarks ===\n")
    print("[1] PyTorch Arabic predictor (CPU):")
    bench_torch()
    print("\n[2] gloss->sentence cache:")
    bench_gloss_cache()
