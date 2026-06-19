# ASL Gloss Syntax, Non-Manual Markers & Gloss-and-Stitch

Sign languages are not word-for-word encodings of spoken language. ASL uses a
**Topic-Comment** structure, drops articles and copulas, and carries much of its
grammar on the face and body (**non-manual markers**). Together models this in
`app/server/gloss.py` (syntax) and `app/server/stitch.py` (out-of-vocabulary
animation).

## Topic-Comment ↔ SVO

Two directions, both few-shot prompted through the provider abstraction
(Gemini by default, offline LLM fallback):

| Direction | Function | Used by |
|-----------|----------|---------|
| gloss → sentence (sign → text) | `gloss_to_sentence(tokens, language)` | `POST /api/translate/sentence` |
| sentence → gloss (text → sign) | `english_to_gloss(sentence, language)` | `POST /api/gloss` |

Examples (English/ASL):

```
"I am going to the store."   →  STORE I GO
"What is your name?"         →  YOUR NAME WHAT
"My mother loves me."        →  MOTHER LOVE ME
```

Both functions degrade gracefully:
- `gloss_to_sentence` returns the raw gloss if no LLM is reachable.
- `english_to_gloss` falls back to a deterministic rule-based gloss (uppercase,
  strip English articles/copulas) so text-to-sign keeps working offline.

Arabic / ArSL uses a parallel set of few-shot exemplars.

## Non-Manual Markers (NMM)

NMMs are facial/body grammar that span manual signs rather than being separate
signs — e.g. raised eyebrows over a yes/no question, furrowed brows for a
wh-question, a headshake for negation, a brow-raise + head-tilt to mark a topic.

`annotate_nmm(gloss_tokens, sentence)` returns a render-ready structure:

```json
{
  "tokens": ["YOUR", "NAME", "WHAT"],
  "sentence_type": "wh_question",
  "markers": {"eyebrows": "furrowed", "head": "forward"},
  "span": [0, 3]
}
```

`POST /api/gloss` includes this under `nmm`. This is a **documented hook**: the
sentence-type heuristics in `detect_sentence_type` can be swapped for a model
later without changing consumers. The avatar renderer is expected to read
`markers` and apply them across `span`. Marker vocabulary lives in
`gloss.NMM_MARKERS`.

## Gloss & Stitch (out-of-vocabulary words)

When a glossed word has no dedicated sign, `stitch.gloss_and_stitch(word, db)`
synthesises an animation by **fingerspelling** — one clip per letter — and
stitching the clips into one smooth sequence. If any required letter sign is
missing it returns `None`, so the word is reported as unrenderable rather than
shown as a broken animation. `POST /api/signs/batch` calls this automatically for
words it cannot match, tagging stitched results with `"source": "stitch"`.

### Why smoothing is needed

Concatenating clips creates a visible jerk at each boundary: the last pose of
clip A rarely matches the first pose of clip B. `stitch_sequences`:

1. inserts a short **linearly-interpolated bridge** between consecutive clips, and
2. runs a **Savitzky-Golay filter** over a window centred on each stitch
   boundary, smoothing the transition while preserving the body of each sign.

Missing landmarks (`NaN` from MediaPipe) are left untouched — smoothing a window
containing `NaN` would corrupt good neighbouring frames. Parameters
(`TRANSITION_FRAMES`, `SMOOTH_WINDOW`, `SMOOTH_POLYORDER`) are tunable per call.

Landmark arrays are `[n_frames, 543, 3]` (MediaPipe Holistic).
