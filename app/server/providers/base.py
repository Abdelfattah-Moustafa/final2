"""Abstract provider interfaces for LLM, TTS, and STT.

The rest of the app depends only on these interfaces, never on a concrete
vendor. Concrete adapters live in sibling modules (gemini.py, ollama.py,
local.py) and are wired together by the factory in __init__.py based on the
LLM_PROVIDER / TTS_PROVIDER / STT_PROVIDER environment variables.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """Raised by an adapter when it cannot fulfil a request.

    The fallback chain catches this to try the next provider in line.
    """


class LLMProvider(ABC):
    name: str = "llm"

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Return the model's text completion for a prompt. May raise ProviderError."""
        raise NotImplementedError

    def available(self) -> bool:
        """Cheap best-effort check that this provider can be used at all."""
        return True


class TTSProvider(ABC):
    name: str = "tts"

    @abstractmethod
    def synthesize(self, text: str, language: str = "english") -> bytes:
        """Return WAV-encoded audio bytes for the text. May raise ProviderError."""
        raise NotImplementedError

    def available(self) -> bool:
        return True


class STTProvider(ABC):
    name: str = "stt"

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav",
                   language: str = "english") -> str:
        """Return the transcription of the audio. May raise ProviderError."""
        raise NotImplementedError

    def available(self) -> bool:
        return True
