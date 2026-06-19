"""Tests for the provider abstraction + fallback chains (no network required)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "server"))

import pytest

import providers
from providers import (
    FallbackLLM,
    FallbackSTT,
    FallbackTTS,
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
    reset_providers,
)
from providers.base import LLMProvider, ProviderError, STTProvider, TTSProvider


# ── Fakes ─────────────────────────────────────────────────────
class _OKLLM(LLMProvider):
    name = "ok"
    def __init__(self, reply="hello"):
        self._reply = reply
    def generate(self, prompt, temperature=0.0):
        return self._reply


class _FailLLM(LLMProvider):
    name = "fail"
    def generate(self, prompt, temperature=0.0):
        raise ProviderError("boom")


class _UnavailLLM(LLMProvider):
    name = "unavail"
    def available(self):
        return False
    def generate(self, prompt, temperature=0.0):
        raise AssertionError("should never be called when unavailable")


# ── Fallback ordering ─────────────────────────────────────────
def test_fallback_uses_primary_when_ok():
    chain = FallbackLLM([_OKLLM("primary"), _OKLLM("secondary")])
    assert chain.generate("hi") == "primary"


def test_fallback_skips_failing_primary():
    chain = FallbackLLM([_FailLLM(), _OKLLM("backup")])
    assert chain.generate("hi") == "backup"


def test_fallback_skips_unavailable_primary():
    chain = FallbackLLM([_UnavailLLM(), _OKLLM("backup")])
    assert chain.generate("hi") == "backup"


def test_fallback_raises_when_all_fail():
    chain = FallbackLLM([_FailLLM(), _FailLLM()])
    with pytest.raises(ProviderError):
        chain.generate("hi")


def test_chain_names_reports_order():
    chain = FallbackLLM([_OKLLM(), _FailLLM()])
    assert chain.chain_names == ["ok", "fail"]


def test_fallback_drops_none_adapters():
    chain = FallbackLLM([None, _OKLLM("only")])
    assert chain.chain_names == ["ok"]
    assert chain.generate("x") == "only"


# ── Factory / env selection ───────────────────────────────────
def test_default_llm_is_gemini_with_ollama_fallback(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK", raising=False)
    reset_providers()
    chain = get_llm_provider()
    assert chain.chain_names == ["gemini", "ollama"]


def test_tts_default_chain(monkeypatch):
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    monkeypatch.delenv("TTS_FALLBACK", raising=False)
    reset_providers()
    assert get_tts_provider().chain_names == ["gemini", "pyttsx3"]


def test_stt_default_chain(monkeypatch):
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("STT_FALLBACK", raising=False)
    reset_providers()
    assert get_stt_provider().chain_names == ["gemini", "whisper"]


def test_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK", "0")
    reset_providers()
    assert get_llm_provider().chain_names == ["gemini"]


def test_explicit_ollama_primary_has_no_duplicate(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_FALLBACK", raising=False)
    reset_providers()
    # ollama is primary; the auto-fallback (also ollama) must not be appended.
    assert get_llm_provider().chain_names == ["ollama"]


def teardown_module(module):
    reset_providers()
