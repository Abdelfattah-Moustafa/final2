"""Tests for ASL gloss transformation + non-manual markers (no network)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "server"))

import gloss
from providers.base import LLMProvider, ProviderError


class _FakeLLM(LLMProvider):
    name = "fake"
    def __init__(self, reply):
        self._reply = reply
    def generate(self, prompt, temperature=0.0):
        return self._reply


class _DeadLLM(LLMProvider):
    name = "dead"
    def generate(self, prompt, temperature=0.0):
        raise ProviderError("no llm")


# ── gloss → sentence ──────────────────────────────────────────
def test_gloss_to_sentence_uses_llm():
    out = gloss.gloss_to_sentence(["STORE", "I", "GO"], llm=_FakeLLM("I am going to the store."))
    assert out == "I am going to the store."


def test_gloss_to_sentence_strips_quotes():
    out = gloss.gloss_to_sentence(["ME", "HAPPY"], llm=_FakeLLM('"I am happy."'))
    assert out == "I am happy."


def test_gloss_to_sentence_falls_back_to_raw_gloss():
    out = gloss.gloss_to_sentence(["STORE", "I", "GO"], llm=_DeadLLM())
    assert out == "STORE I GO"


def test_gloss_to_sentence_empty():
    assert gloss.gloss_to_sentence([], llm=_FakeLLM("x")) == ""


# ── gloss→sentence cache ──────────────────────────────────────
class _CountingLLM(LLMProvider):
    name = "counting"
    def __init__(self):
        self.calls = 0
    def generate(self, prompt, temperature=0.0):
        self.calls += 1
        return "I am going to the store."


def test_explicit_llm_is_not_cached():
    # Passing an llm explicitly must bypass the module cache (keeps tests pure).
    gloss.clear_sentence_cache()
    llm = _CountingLLM()
    gloss.gloss_to_sentence(["STORE", "I", "GO"], llm=llm)
    gloss.gloss_to_sentence(["STORE", "I", "GO"], llm=llm)
    assert llm.calls == 2  # no caching when llm injected


def test_default_provider_path_uses_cache(monkeypatch):
    # When llm is None, results are cached keyed by (gloss, language).
    gloss.clear_sentence_cache()
    llm = _CountingLLM()
    monkeypatch.setattr(gloss, "get_llm_provider", lambda: llm)
    first = gloss.gloss_to_sentence(["STORE", "I", "GO"])
    second = gloss.gloss_to_sentence(["STORE", "I", "GO"])
    assert first == second
    assert llm.calls == 1  # second call served from cache


# ── sentence → gloss ──────────────────────────────────────────
def test_english_to_gloss_uses_llm():
    toks = gloss.english_to_gloss("I am going to the store.", llm=_FakeLLM("STORE I GO"))
    assert toks == ["STORE", "I", "GO"]


def test_english_to_gloss_rule_based_fallback():
    # No LLM: deterministic fallback strips articles/copulas, uppercases.
    toks = gloss.english_to_gloss("I am going to the store", llm=_DeadLLM())
    assert "STORE" in toks
    assert "GOING" in toks
    # 'am', 'to', 'the' are function words and must be dropped.
    assert "AM" not in toks and "TO" not in toks and "THE" not in toks


def test_english_to_gloss_empty():
    assert gloss.english_to_gloss("", llm=_FakeLLM("x")) == []


# ── sentence type detection ───────────────────────────────────
def test_detect_wh_question():
    assert gloss.detect_sentence_type("What is your name?") == "wh_question"


def test_detect_yes_no_question():
    assert gloss.detect_sentence_type("Are you hungry?") == "yes_no_question"


def test_detect_negation():
    assert gloss.detect_sentence_type("I do not want it") == "negation"


def test_detect_statement():
    assert gloss.detect_sentence_type("I am happy") == "statement"


# ── NMM annotation hook ───────────────────────────────────────
def test_annotate_nmm_wh_question():
    ann = gloss.annotate_nmm(["YOUR", "NAME", "WHAT"], "What is your name?")
    assert ann["sentence_type"] == "wh_question"
    assert ann["markers"]["eyebrows"] == "furrowed"
    assert ann["span"] == [0, 3]
    assert ann["tokens"] == ["YOUR", "NAME", "WHAT"]


def test_annotate_nmm_statement_has_no_markers():
    ann = gloss.annotate_nmm(["ME", "HAPPY"], "I am happy")
    assert ann["sentence_type"] == "statement"
    assert ann["markers"] == {}
