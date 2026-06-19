"""Provider factory + fallback chains.

Selection is driven by environment variables:
    LLM_PROVIDER  (gemini | ollama)            default: gemini
    TTS_PROVIDER  (gemini | pyttsx3)           default: gemini
    STT_PROVIDER  (gemini | whisper)           default: gemini

Each primary provider is automatically backed by an offline fallback so the
platform keeps working (degraded) without a cloud key or network:
    LLM  -> Ollama
    TTS  -> pyttsx3
    STT  -> local Whisper

Set the *_FALLBACK env var to "0" to disable a chain's offline fallback.
"""
from __future__ import annotations

import os

from .base import LLMProvider, ProviderError, STTProvider, TTSProvider


# ── Fallback wrappers ─────────────────────────────────────────
class _FallbackMixin:
    def __init__(self, providers: list):
        # Drop Nones; keep order (primary first).
        self._providers = [p for p in providers if p is not None]

    @property
    def chain_names(self) -> list[str]:
        return [p.name for p in self._providers]


class FallbackLLM(_FallbackMixin, LLMProvider):
    name = "fallback-llm"

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        last_err = None
        for p in self._providers:
            try:
                if not p.available():
                    continue
                return p.generate(prompt, temperature)
            except ProviderError as e:
                last_err = e
                continue
        raise ProviderError(f"All LLM providers failed (tried {self.chain_names}): {last_err}")


class FallbackTTS(_FallbackMixin, TTSProvider):
    name = "fallback-tts"

    def synthesize(self, text: str, language: str = "english") -> bytes:
        last_err = None
        for p in self._providers:
            try:
                if not p.available():
                    continue
                return p.synthesize(text, language)
            except ProviderError as e:
                last_err = e
                continue
        raise ProviderError(f"All TTS providers failed (tried {self.chain_names}): {last_err}")


class FallbackSTT(_FallbackMixin, STTProvider):
    name = "fallback-stt"

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav",
                   language: str = "english") -> str:
        last_err = None
        for p in self._providers:
            try:
                if not p.available():
                    continue
                return p.transcribe(audio_bytes, mime_type, language)
            except ProviderError as e:
                last_err = e
                continue
        raise ProviderError(f"All STT providers failed (tried {self.chain_names}): {last_err}")


# ── Adapter construction ──────────────────────────────────────
def _make_llm(name: str):
    if name == "gemini":
        from .gemini import GeminiLLM
        return GeminiLLM()
    if name == "ollama":
        from .ollama import OllamaLLM
        return OllamaLLM()
    return None


def _make_tts(name: str):
    if name == "gemini":
        from .gemini import GeminiTTS
        return GeminiTTS()
    if name == "pyttsx3":
        from .local import Pyttsx3TTS
        return Pyttsx3TTS()
    return None


def _make_stt(name: str):
    if name == "gemini":
        from .gemini import GeminiSTT
        return GeminiSTT()
    if name == "whisper":
        from .local import WhisperSTT
        return WhisperSTT()
    return None


def _fallback_enabled(kind: str) -> bool:
    return os.environ.get(f"{kind}_FALLBACK", "1").strip() != "0"


# ── Cached singletons ─────────────────────────────────────────
_llm = None
_tts = None
_stt = None


def get_llm_provider() -> LLMProvider:
    global _llm
    if _llm is None:
        primary = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
        chain = [_make_llm(primary)]
        if _fallback_enabled("LLM") and primary != "ollama":
            chain.append(_make_llm("ollama"))
        _llm = FallbackLLM(chain)
    return _llm


def get_tts_provider() -> TTSProvider:
    global _tts
    if _tts is None:
        primary = os.environ.get("TTS_PROVIDER", "gemini").strip().lower()
        chain = [_make_tts(primary)]
        if _fallback_enabled("TTS") and primary != "pyttsx3":
            chain.append(_make_tts("pyttsx3"))
        _tts = FallbackTTS(chain)
    return _tts


def get_stt_provider() -> STTProvider:
    global _stt
    if _stt is None:
        primary = os.environ.get("STT_PROVIDER", "gemini").strip().lower()
        chain = [_make_stt(primary)]
        if _fallback_enabled("STT") and primary != "whisper":
            chain.append(_make_stt("whisper"))
        _stt = FallbackSTT(chain)
    return _stt


def reset_providers() -> None:
    """Test hook: clear cached singletons so env changes take effect."""
    global _llm, _tts, _stt
    _llm = _tts = _stt = None


__all__ = [
    "LLMProvider", "TTSProvider", "STTProvider", "ProviderError",
    "get_llm_provider", "get_tts_provider", "get_stt_provider", "reset_providers",
]
