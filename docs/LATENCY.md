# Latency Profiling & Optimization (Phase 5)

This phase instruments the translation pipeline end-to-end and removes the main
sources of latency and event-loop blocking. Numbers below for the PyTorch
predictor and the gloss cache are **measured on CPU** via `scripts/benchmark.py`;
others are characterised qualitatively because they need a camera, GPU, or live
network.

## How to measure

- **Live, per-stage:** every hot stage is wrapped in a `Timer` (see
  `app/server/profiling.py`) and aggregated at **`GET /api/metrics`** —
  `count / avg / p50 / p95 / max / last` ms per stage
  (`infer.asl`, `infer.arabic`, `llm.gloss_to_sentence`, `llm.english_to_gloss`,
  `stt.transcribe`, `signs.*`).
- **Offline micro-benchmarks:** `python scripts/benchmark.py`.

## Optimizations

### 1. Blocking calls moved off the event loop  *(correctness + throughput)*

The async endpoints previously ran blocking work — TFLite/PyTorch inference,
network LLM/STT calls, SBERT encoding — **directly in the event loop**, so a
single in-flight inference serialized every other request (including health
checks and the meeting socket). All such calls now run via
`starlette.concurrency.run_in_threadpool`:

| Endpoint | Blocking work offloaded |
|----------|-------------------------|
| `/api/translate` | TFLite (ASL) / PyTorch (Arabic) inference |
| `/api/translate/sentence` | LLM gloss→sentence |
| `/api/gloss` | LLM sentence→gloss |
| `/api/stt` | network speech-to-text |
| `/api/signs/lookup`, `/api/signs/{word}`, `/api/signs/batch` | SBERT encode + LLM translate + DB |

Impact: under concurrent load the event loop stays responsive; p95 of unrelated
requests no longer tracks the slowest inference. (`/api/tts` was already a sync
`def`, which Starlette runs in a threadpool automatically.)

### 2. PyTorch Arabic predictor — `inference_mode` + INT8  *(measured)*

- `torch.no_grad()` → `torch.inference_mode()` (drops version-counter/view
  tracking overhead).
- INT8 **dynamic quantization** of `Linear` + `GRU` weights on CPU
  (`TORCH_INT8=1`, default on; opt out with `TORCH_INT8=0`).

Measured on CPU, synthetic `[60, 543, 3]` input, 50 runs:

```
fp32 (inference_mode)  :  7.48 ms / inference
int8 dynamic           :  4.21 ms / inference
-> 1.78x faster (7.48 -> 4.21 ms)
```

### 3. TFLite — XNNPACK + multi-threading

The LiteRT interpreter is now constructed with
`num_threads = min(4, cores)` (override via `TFLITE_NUM_THREADS`). LiteRT applies
the **XNNPACK** delegate by default for float CPU models, accelerating the
conv/dense ops. Falls back gracefully on older interpreter signatures.

### 4. gloss→sentence cache  *(measured)*

The same recognized gloss recurs constantly during a conversation. A bounded
(256-entry) cache keyed on `(gloss, language)` short-circuits repeat LLM
round-trips:

```
cold (50 ms simulated LLM)  : 50.13 ms
warm (cache hit)            :  0.001 ms
-> effectively eliminates repeat calls
```

In production this removes a full Gemini round-trip (typically 300–800 ms) for
any repeated gloss.

### 5. Sliding buffer

Inference already operates on a bounded rolling window: the ASL TFLite path trims
to `SEQUENCE_LENGTH = 60` frames and the Arabic path samples fixed `[30, 45, 60]`
windows, so payload growth does not inflate inference cost.

## Net effect

- The dominant cost in **sign→text** is the LLM gloss→sentence call; it is now
  cached (repeat) and threadpool-offloaded (cold), so it no longer blocks other
  requests.
- **Arabic recognition** inference is ~1.8× faster on CPU.
- The server stays responsive under concurrency because no endpoint blocks the
  event loop.
- `GET /api/metrics` makes regressions visible without external tooling.
