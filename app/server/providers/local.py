"""Local offline adapters: pyttsx3 (TTS) and faster-whisper / openai-whisper (STT).

These run fully offline so the platform degrades gracefully when no cloud key is
configured or the network is down. They are heavier per-call and lower quality
than Gemini, so they sit at the end of the fallback chain.
"""
from __future__ import annotations

import os
import tempfile
from threading import Lock

from .base import ProviderError, STTProvider, TTSProvider

_pyttsx3_lock = Lock()


class Pyttsx3TTS(TTSProvider):
    name = "pyttsx3"

    def available(self) -> bool:
        try:
            import pyttsx3  # noqa: F401
            return True
        except Exception:
            return False

    def synthesize(self, text: str, language: str = "english") -> bytes:
        try:
            import pyttsx3
        except ImportError as e:
            raise ProviderError("pyttsx3 is not installed") from e

        with _pyttsx3_lock:
            try:
                engine = pyttsx3.init()
            except Exception as e:  # noqa: BLE001
                raise ProviderError(f"pyttsx3 init failed: {e}") from e

            # Prefer a voice matching the requested language when available.
            wants_arabic = language in ("arabic", "ar", "egyptian", "eg")
            try:
                voices = engine.getProperty("voices")
                for v in voices:
                    name_l, id_l = v.name.lower(), v.id.lower()
                    is_ar = ("arabic" in name_l or "ar-" in id_l or "ar_" in id_l
                             or "naayf" in name_l or "hoda" in name_l)
                    if wants_arabic and is_ar:
                        engine.setProperty("voice", v.id)
                        break
            except Exception:
                pass  # fall back to default voice

            tmp = os.path.join(tempfile.gettempdir(), f"pyttsx3_{os.getpid()}.wav")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            try:
                engine.save_to_file(text, tmp)
                engine.runAndWait()
                if not os.path.exists(tmp):
                    raise ProviderError("pyttsx3 produced no WAV file")
                with open(tmp, "rb") as f:
                    return f.read()
            except ProviderError:
                raise
            except Exception as e:  # noqa: BLE001
                raise ProviderError(f"pyttsx3 synthesis failed: {e}") from e
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass


class WhisperSTT(STTProvider):
    """Local Whisper transcription via faster-whisper (preferred) or openai-whisper."""

    name = "whisper"

    def __init__(self) -> None:
        self.model_size = os.environ.get("WHISPER_MODEL", "base")
        self._model = None
        self._backend = None
        self._lock = Lock()

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            try:
                import whisper  # noqa: F401
                return True
            except Exception:
                return False

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                self._backend = "faster"
                return
            except Exception:
                pass
            try:
                import whisper
                self._model = whisper.load_model(self.model_size)
                self._backend = "openai"
            except Exception as e:  # noqa: BLE001
                raise ProviderError(f"No local Whisper backend available: {e}") from e

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav",
                   language: str = "english") -> str:
        self._ensure_model()
        lang_code = "ar" if language in ("arabic", "ar", "egyptian", "eg") else "en"

        suffix = ".wav"
        if "ogg" in mime_type:
            suffix = ".ogg"
        elif "mp3" in mime_type or "mpeg" in mime_type:
            suffix = ".mp3"
        elif "webm" in mime_type:
            suffix = ".webm"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()
            if self._backend == "faster":
                segments, _ = self._model.transcribe(tmp.name, language=lang_code)
                return " ".join(s.text for s in segments).strip()
            result = self._model.transcribe(tmp.name, language=lang_code)
            return (result.get("text") or "").strip()
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Whisper transcription failed: {e}") from e
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
