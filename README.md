# Together

**Real-time, bi-directional sign-language translation in the browser.**

Together translates **American Sign Language (ASL)** and **Arabic / Egyptian
Sign Language (ArSL)** to text and speech — and turns speech and text back into
signs — live, with a webcam, no install. The UI is fully **bilingual**
(English LTR + Arabic RTL) and ships a light/dark themed design system.

> Built for and with the Deaf and Hard-of-Hearing community: sign → text,
> sign → speech, text → sign, speech → sign, and a live two-person **meeting**
> mode that translates between a signer and a speaker over WebRTC.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [The translation pipelines](#the-translation-pipelines)
- [Frontend & design system](#frontend--design-system)
- [Project layout](#project-layout)
- [Running locally](#running-locally)
- [Configuration](#configuration)
- [Testing](#testing)
- [API surface](#api-surface)

---

## What it does

| Module | Direction | How |
| --- | --- | --- |
| **Sign → Text** | camera → text | MediaPipe Holistic landmarks → sign model → gloss → LLM forms a sentence |
| **Sign → Speech** | camera → audio | as above, then TTS speaks the sentence |
| **Text → Sign** | text → avatar | sentence → `/api/gloss` (Topic-Comment ordering) → landmark lookup → avatar |
| **Speech → Sign** | mic → avatar | Web Speech / STT → text → same text-to-sign path |
| **Live Meeting** | signer ↔ speaker | WebRTC video + Socket.IO signaling, live captions / sign avatar per role |

Two languages are first-class: **English/ASL** and **Arabic/ArSL** (with an
Egyptian variant), each with its own model and its own dashboard.

## Architecture

Four decisions anchor the project:

1. **Frontend is server-rendered HTML/CSS/JS** — Jinja2 templates, a vanilla
   design-token system, no SPA framework or build step.
2. **ML stack**: TFLite (`ai-edge-litert`) for the 250-class English ASL model;
   a PyTorch CNN-GRU for the 20-class Arabic model. MediaPipe Holistic extracts
   pose/hand/face landmarks in the browser.
3. **Real-time comms**: Socket.IO for WebRTC signaling (meeting mode).
4. **Both languages**, with JWT auth (Argon2 hashing, bcrypt-legacy verify,
   httpOnly refresh tokens, sliding-window rate limiting).

```
Browser (MediaPipe Holistic, WebRTC, vanilla JS)
   │  landmarks / text / audio        WebRTC media ⇄ peer
   ▼                                   ▲
FastAPI + Socket.IO  ──►  sign models (TFLite / PyTorch)
   │                       SBERT semantic sign lookup
   │
   ├─ Providers (pluggable chains, cloud → offline fallback)
   │     LLM: gemini → ollama     TTS: gemini → pyttsx3     STT: gemini → whisper
   │
   └─ Postgres + pgvector  (users, refresh tokens, sign metadata + embeddings)
```

Provider chains degrade gracefully: if the cloud provider is unavailable the
next link (local Ollama / pyttsx3 / faster-whisper) takes over, and gloss
formation falls back to the raw gloss when no LLM is reachable.

## The translation pipelines

**Sign → Text/Speech.** The browser runs MediaPipe Holistic and streams
landmark frames; the server model predicts sign glosses, a vote/cooldown buffer
debounces them, and `/api/translate/sentence` asks the LLM to turn the
Topic-Comment gloss into a natural sentence (see `docs/GLOSS_AND_NMM.md`).

**Text/Speech → Sign.** `/api/gloss` converts a natural sentence into
ASL/ArSL gloss tokens (Topic-Comment order + non-manual markers), then
`/api/signs/batch` (or `/api/signs_ar/batch`) looks up landmark sequences for
each token via SBERT-backed semantic matching, and the avatar plays them. If
`/api/gloss` is unavailable the client falls back to naive word tokenization.

Latency work (model warmup, batching, caching) is documented in
`docs/LATENCY.md`; provider selection and cost in
`docs/PROVIDERS_AND_PRICING.md`.

## Frontend & design system

The UI is a small **design-token system** so a restyle is a one-file change and
light/dark/RTL come for free:

- **`app/static/css/theme.css`** — the single source of truth: color, type,
  spacing, elevation tokens. Light/dark are token swaps via
  `html[data-theme="light|dark"]`, with a no-JS `prefers-color-scheme`
  fallback. Brand fonts (**Thmanyah** Sans + Serif-Display) are bundled
  locally — no CDN dependency.
- **`app/static/css/ui.css`** — reusable components, all prefixed `t-`
  (`.t-btn`, `.t-card`, `.t-pill`, `.t-sidebar`, `.t-toggle`, …). RTL is handled
  with CSS **logical properties** (`inset-inline-*`), so Arabic mirrors at zero
  cost. Includes focus-visible rings, a skip-link, and
  `prefers-reduced-motion` handling.
- **`app/static/js/theme.js`** — a no-flash theme controller: applies the
  stored/OS theme synchronously before paint and exposes `window.Theme`.

Dashboard-specific styles live under the `d-` prefix inside the templates. The
English (`index.html`) and Arabic (`index_ar.html`) dashboards share the same
structure and inline JS; the Arabic one is RTL, forces the ArSL model, and bakes
Arabic copy directly into the markup.

## Project layout

```
app/
  server/
    main.py            FastAPI app, routes, Socket.IO server, startup
    auth.py            JWT, Argon2/bcrypt hashing, rate limiting
    asl_service.py     SignDB / ArabicSignDB / ASLService (SBERT lookup)
    gloss.py           gloss ⇄ sentence + non-manual markers
    providers/         pluggable LLM / TTS / STT chains
    db/                SQLAlchemy models, pgvector, repository
    alembic/           database migrations
  templates/           landing, login, signup, index (EN), index_ar (AR)
  static/css/          theme.css (tokens), ui.css (components), main.css (legacy)
  static/js/           app.js, auth.js, theme.js, socket.io.js
  static/fonts/        bundled Thmanyah Sans + Serif-Display (woff2)
tests/                 pytest suite (auth, gloss, providers, stitch, templates)
docs/                  GLOSS_AND_NMM, LATENCY, PROVIDERS_AND_PRICING
docker-compose.yml     postgres (pgvector) + web + ollama
```

## Running locally

### Docker Compose (recommended)

Brings up Postgres (pgvector), the web app, and Ollama:

```bash
cp .env.example .env        # add GEMINI_API_KEY / JWT_SECRET_KEY, etc.
docker compose up --build
# → http://localhost:8000
```

On first run, pull a local LLM for the offline fallback:

```bash
docker exec -it <ollama-container> ollama run llama3.2
```

### Manual

Requires Python 3.11+ and a Postgres 16 with the `vector` extension.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL="postgresql+psycopg://together:together@localhost:5432/together"
export JWT_SECRET_KEY="change-me"
alembic upgrade head          # create schema (pgvector extension + tables)

python start_server.py        # or: PORT=8000 python app/server/main.py
```

`start_server.py` sets thread/ML env guards before importing torch and starts
uvicorn on the Socket.IO ASGI app.

## Configuration

Copy `.env.example` to `.env`. Common variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | Postgres + pgvector DSN | `postgresql+psycopg://together:together@localhost:5432/together` |
| `JWT_SECRET_KEY` | token signing secret | `dev-insecure-change-me` (override!) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` / `REFRESH_TOKEN_EXPIRE_DAYS` | token lifetimes | 60 / 7 |
| `LLM_PROVIDER` / `TTS_PROVIDER` / `STT_PROVIDER` | force a provider | auto (cloud → offline) |
| `GEMINI_API_KEY`, `GEMINI_LLM_MODEL`, `GEMINI_TTS_MODEL`, `GEMINI_STT_MODEL` | Gemini | — |
| `OLLAMA_HOST`, `OLLAMA_MODEL`, `WHISPER_MODEL` | offline fallbacks | — |
| `ALLOWED_ORIGINS` | CORS allow-list | localhost |
| `TFLITE_NUM_THREADS`, `DB_POOL_SIZE`, `SLOW_REQUEST_MS` | tuning | — |

## Testing

```bash
pip install pytest httpx
pytest tests/ -q
```

The suite covers auth (hashing, tokens, rate limiting), gloss/NMM, providers,
sentence stitching, profiling, and a **frontend regression** module
(`tests/test_templates.py`) that renders every template and asserts the
invariants that broke during the redesign — the dashboards keep their
WASM/MediaPipe/socket.io `{% block head %}`, the design-system stylesheets stay
linked, the legacy JS hooks exist, every `getElementById` target resolves, the
Arabic dashboard is RTL with no English leaks, and the inline JS parses
(`node --check`, skipped if node is absent).

## API surface

| Method & path | Purpose |
| --- | --- |
| `GET /`, `/login`, `/signup`, `/dashboard?lang=en\|ar` | pages |
| `POST /api/auth/signup` · `login` · `refresh` · `logout` · `GET me` | auth |
| `POST /api/translate` · `translate/sentence` | sign gloss → sentence |
| `POST /api/gloss` | sentence → ASL/ArSL gloss + non-manual markers |
| `GET /api/tts` · `POST /api/stt` | speech synthesis / recognition |
| `GET /api/signs/lookup` · `/{word}` · `POST /api/signs/batch` · `/api/signs_ar/batch` | sign landmark lookup |
| `GET /api/health` · `/api/metrics` | health & profiling |
| Socket.IO: `join_room`, `leave_room`, `announce_presence`, `webrtc_offer/answer/ice_candidate`, `translate_sentence` | meeting signaling |

---

_Sign language, bridged in real time._
