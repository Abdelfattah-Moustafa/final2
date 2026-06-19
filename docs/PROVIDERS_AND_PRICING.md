# Providers & API Pricing

Together routes every LLM, text-to-speech (TTS), and speech-to-text (STT) call
through a pluggable provider abstraction (`app/server/providers/`). You pick the
primary provider per capability with environment variables; an offline fallback
runs automatically when no key is set or a cloud call fails.

```
LLM_PROVIDER = gemini | ollama          (fallback: ollama)
TTS_PROVIDER = gemini | pyttsx3         (fallback: pyttsx3)
STT_PROVIDER = gemini | whisper         (fallback: local Whisper)
```

Set `LLM_FALLBACK=0` / `TTS_FALLBACK=0` / `STT_FALLBACK=0` to disable a chain's
offline fallback.

---

## What you actually need to subscribe to

The **only** paid API the default configuration uses is **Google Gemini**. The
fallbacks (Ollama, pyttsx3, Whisper) are free and run locally — no subscription.

| Capability | Default (paid) | Offline fallback (free) |
|------------|----------------|--------------------------|
| LLM (gloss → sentence, translation) | Google Gemini | Ollama (`llama3.2`) |
| TTS (speech output) | Google Gemini TTS | pyttsx3 (OS voices) |
| STT (speech input) | Google Gemini | local Whisper |

---

## Pricing

> ⚠️ Prices below are approximate and change frequently. **Verify against the
> vendor's live pricing page before committing to a paid plan.** Figures are
> current to early 2026.

### Google Gemini (default) — https://ai.google.dev/pricing

- **Free tier:** Generous rate-limited free quota on the `*-flash` models —
  enough for development and light demo usage. This is what the project ships
  with today (the free-tier key was the one previously hardcoded; it now comes
  from `GEMINI_API_KEY` in `.env`).
- **Paid (pay-as-you-go), approximate:**
  - `gemini-2.5-flash` (LLM): ~\$0.075 / 1M input tokens, ~\$0.30 / 1M output tokens.
  - `gemini-2.5-flash` TTS / audio: billed per audio second / token; budget a
    few dollars per hour of synthesized or transcribed audio.
- **When to upgrade:** Move from the free key to a paid Google Cloud billing
  account when you hit rate limits or need a production SLA. Only the value of
  `GEMINI_API_KEY` changes — no code change.

### Optional cloud alternatives (not used by default)

These adapters are scaffolded for via `LLM_PROVIDER` but require their own keys
and a small adapter addition if you choose to enable them later:

- **OpenAI** — https://openai.com/api/pricing/ (`OPENAI_API_KEY`)
- **Anthropic Claude** — https://www.anthropic.com/pricing (`ANTHROPIC_API_KEY`)

### Free / local (no subscription)

- **Ollama** — https://ollama.com — runs LLMs locally. Cost = your own compute.
  First run: `ollama run llama3.2`. Configured via `OLLAMA_HOST` / `OLLAMA_MODEL`.
- **pyttsx3** — offline TTS using the operating system's installed voices. Quality
  depends on the OS; install an Arabic voice for best Arabic output.
- **Whisper** (`faster-whisper`) — offline STT. `WHISPER_MODEL` picks the size
  (`tiny`…`large`); larger = more accurate but slower and more memory.

---

## Recommended setups

| Scenario | Config |
|----------|--------|
| **Development / demo** | Gemini free-tier key in `GEMINI_API_KEY`. Fallbacks on. |
| **Production, cloud** | Paid Gemini key; consider `*_FALLBACK=0` if you want hard failures instead of degraded local output. |
| **Fully offline / air-gapped** | `LLM_PROVIDER=ollama`, `TTS_PROVIDER=pyttsx3`, `STT_PROVIDER=whisper`; leave `GEMINI_API_KEY` blank. |
