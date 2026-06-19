"""Ollama LLM adapter — local, offline fallback for the LLM provider."""
from __future__ import annotations

import os

import requests

from .base import LLMProvider, ProviderError


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self) -> None:
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=60,
            )
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Ollama LLM failed: {e}") from e
