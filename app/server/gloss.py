"""ASL / ArSL gloss syntax.

Sign languages do not follow the spoken-language word order. ASL favours a
**Topic-Comment** structure (often Object/Topic first, then subject + verb),
drops copulas/articles, and carries grammar on the face and body
(*non-manual markers*, NMM). This module converts in both directions:

    SVO sentence            <->            ASL gloss (Topic-Comment)
    "I am going to the store"   <->   STORE I GO

Translation uses few-shot prompting through the provider abstraction (Gemini by
default, offline LLM fallback). The non-manual-marker layer is a documented hook
(`annotate_nmm`) the avatar renderer can later consume to drive facial grammar.
"""
from __future__ import annotations

import re
from typing import Optional

from providers import ProviderError, get_llm_provider

# ── Few-shot exemplars ────────────────────────────────────────
# Kept compact and high-signal; the LLM generalises from these.

_GLOSS_TO_SENTENCE_EN = """\
You translate ASL gloss (Topic-Comment order, no articles/copula) into one
natural English sentence. Every signed concept must appear. Add only function
words (articles, copulas, prepositions, pronouns). Output ONLY the sentence.

Gloss: STORE I GO
Sentence: I am going to the store.

Gloss: FINISH EAT I
Sentence: I have finished eating.

Gloss: YOUR NAME WHAT
Sentence: What is your name?

Gloss: ME HAPPY
Sentence: I am happy.

Gloss: MOTHER LOVE ME
Sentence: My mother loves me.
"""

_SENTENCE_TO_GLOSS_EN = """\
You convert an English sentence into ASL gloss using Topic-Comment order.
Rules: UPPERCASE tokens; drop articles (a/an/the) and copulas (is/are/am);
move the topic (often the object/place/time) to the front; keep one concept per
token; use ME for first person object, YOU for second person. Append a question
word at the end for wh-questions. Output ONLY the gloss tokens separated by
single spaces.

Sentence: I am going to the store.
Gloss: STORE I GO

Sentence: What is your name?
Gloss: YOUR NAME WHAT

Sentence: I am happy.
Gloss: ME HAPPY

Sentence: My mother loves me.
Gloss: MOTHER LOVE ME

Sentence: Do you want to eat?
Gloss: YOU WANT EAT YOU
"""

_GLOSS_TO_SENTENCE_AR = """\
حوّل كلمات لغة الإشارة العربية (ترتيب الموضوع ثم التعليق، بدون أدوات أو أفعال ربط)
إلى جملة عربية فصحى طبيعية واحدة. يجب أن يظهر معنى كل إشارة. أضف فقط الأدوات وحروف
الجر والروابط اللازمة. اكتب الجملة فقط.

الإشارات: البيت أنا ذهب
الجملة: أنا ذاهب إلى البيت.

الإشارات: أم حب أنا
الجملة: أمي تحبني.

الإشارات: أنت اسم ماذا
الجملة: ما اسمك؟
"""

_SENTENCE_TO_GLOSS_AR = """\
حوّل الجملة العربية إلى إشارات بترتيب الموضوع ثم التعليق. القواعد: احذف أدوات
التعريف وأفعال الربط؛ قدّم الموضوع (غالباً المفعول/المكان/الزمان)؛ مفهوم واحد لكل
كلمة. اكتب الإشارات فقط مفصولة بمسافة.

الجملة: أنا ذاهب إلى البيت.
الإشارات: البيت أنا ذهب

الجملة: ما اسمك؟
الإشارات: أنت اسم ماذا

الجملة: أمي تحبني.
الإشارات: أم حب أنا
"""


def _clean(text: str) -> str:
    return text.replace('"', "").replace("'", "").strip()


# Bounded cache for gloss→sentence results. The same recognized gloss recurs
# constantly during a conversation, so caching avoids repeat LLM round-trips
# (the dominant latency in the sign→text path). Only used when the default
# provider is in play (llm arg is None) to keep tests/injected providers pure.
_SENTENCE_CACHE: dict = {}
_SENTENCE_CACHE_MAX = 256


def clear_sentence_cache() -> None:
    _SENTENCE_CACHE.clear()


def _is_arabic(language: str) -> bool:
    return (language or "").lower().strip() in ("arabic", "ar", "egyptian", "eg")


# ── Public API ────────────────────────────────────────────────
def gloss_to_sentence(
    gloss_tokens: list[str],
    language: str = "english",
    llm=None,
) -> str:
    """Topic-Comment gloss → natural SVO sentence (few-shot).

    Returns the original gloss joined by spaces if the LLM is unavailable, so the
    caller always gets *something* renderable.
    """
    use_cache = llm is None
    llm = llm or get_llm_provider()
    gloss = " ".join(t for t in gloss_tokens if t)
    if not gloss:
        return ""

    cache_key = (gloss, (language or "english").lower().strip())
    if use_cache and cache_key in _SENTENCE_CACHE:
        return _SENTENCE_CACHE[cache_key]

    preamble = _GLOSS_TO_SENTENCE_AR if _is_arabic(language) else _GLOSS_TO_SENTENCE_EN
    label_in, label_out = (("الإشارات", "الجملة") if _is_arabic(language)
                           else ("Gloss", "Sentence"))
    prompt = f"{preamble}\n{label_in}: {gloss}\n{label_out}:"

    try:
        out = llm.generate(prompt, temperature=0.2)
        out = _clean(out) or gloss
    except ProviderError:
        return gloss  # don't cache transient failures

    if use_cache:
        if len(_SENTENCE_CACHE) >= _SENTENCE_CACHE_MAX:
            _SENTENCE_CACHE.pop(next(iter(_SENTENCE_CACHE)))
        _SENTENCE_CACHE[cache_key] = out
    return out


def english_to_gloss(
    sentence: str,
    language: str = "english",
    llm=None,
) -> list[str]:
    """SVO sentence → ASL/ArSL gloss tokens (Topic-Comment, few-shot).

    Falls back to a deterministic rule-based gloss (uppercase, strip
    articles/copulas) if the LLM is unavailable, so text-to-sign still works
    offline.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return []

    llm = llm or get_llm_provider()
    preamble = _SENTENCE_TO_GLOSS_AR if _is_arabic(language) else _SENTENCE_TO_GLOSS_EN
    label_in, label_out = (("الجملة", "الإشارات") if _is_arabic(language)
                           else ("Sentence", "Gloss"))
    prompt = f"{preamble}\n{label_in}: {sentence}\n{label_out}:"

    try:
        out = llm.generate(prompt, temperature=0.2)
        out = _clean(out)
        tokens = [t for t in re.split(r"\s+", out) if t]
        if tokens:
            return tokens
    except ProviderError:
        pass

    return _rule_based_gloss(sentence, language)


# Function words removed in the offline rule-based gloss fallback (English).
_EN_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "be", "been", "being",
    "to", "of", "for", "and", "or", "do", "does", "did",
}


def _rule_based_gloss(sentence: str, language: str) -> list[str]:
    """Deterministic offline fallback: strip function words, uppercase.

    Arabic keeps every token (no reliable offline stopword list here); English
    drops articles/copulas. This is a coarse approximation of Topic-Comment — it
    does not reorder — but keeps the pipeline functional without an LLM.
    """
    words = re.findall(r"[^\s]+", sentence)
    cleaned = []
    for w in words:
        tok = re.sub(r"[^\w']", "", w)
        if not tok:
            continue
        if not _is_arabic(language) and tok.lower() in _EN_STOPWORDS:
            continue
        cleaned.append(tok.upper() if not _is_arabic(language) else tok)
    return cleaned


# ── Non-manual markers (documented hook) ──────────────────────
# Non-manual markers are the facial/body grammar of sign languages. They are not
# extra signs — they co-occur with manual signs over a span. The avatar renderer
# is expected to consume these annotations to drive eyebrows, head, and mouth.
#
# This is a deliberately small, rule-based first cut; a future model can replace
# the heuristics in `detect_sentence_type` without changing the consumers.

NMM_MARKERS = {
    "yes_no_question": {"eyebrows": "raised", "head": "forward", "hold": True},
    "wh_question": {"eyebrows": "furrowed", "head": "forward"},
    "negation": {"head": "shake", "mouth": "frown"},
    "topic": {"eyebrows": "raised", "head": "tilt"},
    "statement": {},
}

_WH_WORDS = {"what", "where", "when", "who", "why", "how", "which",
             "ماذا", "أين", "متى", "من", "لماذا", "كيف", "ما"}
_NEG_WORDS = {"not", "no", "never", "dont", "cant", "wont",
              "لا", "ليس", "لن", "لم", "ما"}


def detect_sentence_type(sentence: str) -> str:
    """Classify a source sentence so the right NMM span can be attached."""
    s = (sentence or "").strip().lower()
    if not s:
        return "statement"
    tokens = set(re.findall(r"[^\s]+", re.sub(r"[^\w'\s؀-ۿ]", "", s)))
    if tokens & _WH_WORDS:
        return "wh_question"
    if s.endswith("?"):
        return "yes_no_question"
    if tokens & _NEG_WORDS:
        return "negation"
    return "statement"


def annotate_nmm(gloss_tokens: list[str], sentence: str) -> dict:
    """Attach non-manual-marker metadata spanning the gloss.

    Returns a structure the avatar layer can render:
        {
          "tokens": [...],
          "sentence_type": "wh_question",
          "markers": {"eyebrows": "furrowed", "head": "forward"},
          "span": [0, len(tokens)],   # marker applies across this token range
        }
    The hook is intentionally simple and side-effect free; richer per-token NMM
    timing can be layered on later without touching callers.
    """
    stype = detect_sentence_type(sentence)
    return {
        "tokens": list(gloss_tokens),
        "sentence_type": stype,
        "markers": dict(NMM_MARKERS.get(stype, {})),
        "span": [0, len(gloss_tokens)],
    }
