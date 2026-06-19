"""Google Gemini adapters (current default for LLM, TTS, STT).

Ported from the original inline call_gemini_* helpers in main.py. Model names
are read from the environment so a paid Google Cloud key / different model can
be swapped in without code changes (see .env.example).
"""
from __future__ import annotations

import base64
import io
import os
import wave

import requests

from .base import LLMProvider, ProviderError, STTProvider, TTSProvider

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set")
    return key


class GeminiLLM(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        self.model = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        url = f"{_BASE}/{self.model}:generateContent?key={_api_key()}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "topP": 0.95},
        }
        try:
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:  # noqa: BLE001 - normalize to ProviderError for the chain
            raise ProviderError(f"Gemini LLM failed: {e}") from e


class GeminiTTS(TTSProvider):
    name = "gemini"

    def __init__(self) -> None:
        self.model = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    def synthesize(self, text: str, language: str = "english") -> bytes:
        if language in ("egyptian", "eg"):
            lang_name = "Egyptian Arabic dialect"
        elif language in ("english", "en"):
            lang_name = "American English"
        else:
            lang_name = "Modern Standard Arabic"
        prompt = (
            f"Please read the following text aloud. Pronounce it naturally as a native "
            f"speaker of {lang_name}. Output ONLY the audio representation of this text, "
            f"nothing else. Text: {text}"
        )
        url = f"{_BASE}/{self.model}:generateContent?key={_api_key()}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}}
                },
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=20)
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            for part in parts:
                if "inlineData" in part:
                    raw_pcm = base64.b64decode(part["inlineData"]["data"])
                    wav_buf = io.BytesIO()
                    with wave.open(wav_buf, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)       # 16-bit
                        w.setframerate(24000)    # 24 kHz
                        w.writeframes(raw_pcm)
                    return wav_buf.getvalue()
            raise ProviderError("No inlineData in Gemini TTS response")
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Gemini TTS failed: {e}") from e


class GeminiSTT(STTProvider):
    name = "gemini"

    def __init__(self) -> None:
        self.model = os.environ.get("GEMINI_STT_MODEL", "gemini-2.5-flash")

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav",
                   language: str = "english") -> str:
        lang_name = ("Egyptian Arabic dialect" if language in ("egyptian", "eg")
                     else "Modern Standard Arabic" if language in ("arabic", "ar")
                     else "English")
        prompt = (
            f"Transcribe this audio file. The audio is spoken in {lang_name}. "
            f"Return ONLY the exact transcription, without any introductory text, "
            f"explanation, or notes. If there is no speech or it's unintelligible, "
            f"return an empty string."
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        url = f"{_BASE}/{self.model}:generateContent?key={_api_key()}"
        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": mime_type, "data": audio_b64}},
                    {"text": prompt},
                ]
            }]
        }
        try:
            r = requests.post(url, json=payload, timeout=25)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if len(lines) >= 3:
                    text = "\n".join(lines[1:-1]).strip()
            return text
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Gemini STT failed: {e}") from e
