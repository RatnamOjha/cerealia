"""
Government-scheme advisory chatbot.

Design note: the chatbot is *grounded*, not free-running. Every question first
retrieves matching schemes from our own curated dataset, and only those schemes
are passed to the language model as context. This matters here more than in a
typical chatbot -- a hallucinated subsidy percentage or eligibility rule could
cost a farmer real money, so the model is constrained to summarising verified
text rather than recalling scheme details from its training data.

If no API key is configured, or the API call fails, the same retrieved schemes
are rendered directly as a structured answer. The feature degrades to a
retrieval system rather than going down, which also means the demo works with
no network connection.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from . import config  # noqa: F401  -- importing loads backend/.env into os.environ

DATA_DIR = Path(__file__).resolve().parent / "data"

# xAI publishes the key under either name in its own examples; accept both so a
# key pasted from their docs works without the farmer-facing app silently
# staying offline.
GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")
GROK_KEY = (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()
REQUEST_TIMEOUT = float(os.getenv("GROK_TIMEOUT", "25"))

SYSTEM_PROMPT = """You are KrishiMitra, an agricultural advisor for Indian farmers.

Rules you must follow:
1. Answer ONLY from the scheme information provided in the context below. If the
   context does not contain the answer, say so plainly and point the farmer to
   their block agriculture office or Krishi Vigyan Kendra.
2. Never invent subsidy percentages, rupee amounts, deadlines or eligibility
   rules. Wrong numbers cost farmers money. Quote figures exactly as given.
3. Write simply and concretely. Short sentences. Prefer "you get Rs 6,000 a year
   in three instalments" over administrative phrasing.
4. Always end with the concrete next step - which portal, which office, which
   document to carry.
5. Keep answers under 200 words unless the farmer asks for detail."""

HINDI_INSTRUCTION = """
6. REPLY ENTIRELY IN HINDI, in Devanagari script. This is not optional.
   The farmer asked in Hindi and may be listening to the answer read aloud, so
   use plain spoken Hindi rather than formal officialese - say "फसल बीमा", not
   "सस्य बीमा". Keep scheme names and web addresses in their official form:
   pmfby.gov.in stays pmfby.gov.in. Write rupee amounts and dates in digits so
   they are read out correctly.
   The context below carries Hindi wording for each scheme - prefer it."""

# Detecting the script is sufficient here: a farmer writing or speaking Hindi
# produces Devanagari, and nothing else in this application does.
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


@lru_cache(maxsize=1)
def _schemes() -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / "schemes.json").read_text())["schemes"]


def _tokenise(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def detect_language(text: str) -> str:
    """'hi' when the text contains Devanagari, otherwise 'en'."""
    return "hi" if DEVANAGARI.search(text or "") else "en"


def _field(scheme: dict[str, Any], key: str, lang: str) -> str:
    """Hindi variant of a field when asked for and available, else English."""
    if lang == "hi":
        return scheme.get(f"{key}_hi") or scheme.get(key, "")
    return scheme.get(key, "")


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Score every scheme against the query and return the best k.

    A keyword overlap score is enough at this corpus size (12 schemes) and keeps
    the whole thing dependency-free and instant. Swapping in sentence-embedding
    retrieval becomes worthwhile once the corpus reaches hundreds of documents.
    """
    q = _tokenise(query)
    lowered = query.lower()

    scored = []
    for scheme in _schemes():
        # Hindi keywords match as substrings rather than whole tokens: Devanagari
        # inflects with suffixes ("बीमा" inside "बीमे का"), so token equality
        # would miss most natural phrasing.
        keyword_hits = sum(2.0 for kw in scheme["keywords"] if kw in lowered)
        keyword_hits += sum(2.5 for kw in scheme.get("keywords_hi", []) if kw in query)
        name_hits = len(q & _tokenise(scheme["name"])) * 1.5
        body = " ".join([scheme["benefit"], scheme["eligibility"], scheme["category"]])
        body_hits = len(q & _tokenise(body)) * 0.4
        total = keyword_hits + name_hits + body_hits
        if total > 0:
            scored.append((total, scheme))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [s for _, s in scored[:k]]


def _context_block(schemes: list[dict[str, Any]], lang: str = "en") -> str:
    parts = []
    for s in schemes:
        block = (
            f"SCHEME: {s['name']}\n"
            f"Category: {s['category']}\n"
            f"Benefit: {s['benefit']}\n"
            f"Eligibility: {s['eligibility']}\n"
            f"How to apply: {s['how_to_apply']}\n"
            f"Official portal: {s['portal']}"
        )
        if lang == "hi":
            block += (
                "\n--- Hindi wording to prefer ---\n"
                f"नाम: {_field(s, 'name', 'hi')}\n"
                f"लाभ: {_field(s, 'benefit', 'hi')}\n"
                f"पात्रता: {_field(s, 'eligibility', 'hi')}\n"
                f"आवेदन: {_field(s, 'how_to_apply', 'hi')}"
            )
        parts.append(block)
    return "\n\n---\n\n".join(parts)


OFFLINE_LABELS = {
    "en": {
        "none": ("I could not match your question to a scheme in my records. Please "
                 "contact your block agriculture office or the nearest Krishi Vigyan "
                 "Kendra — they can advise on schemes specific to your district.\n\n"
                 "You can also browse every central scheme at https://www.myscheme.gov.in"),
        "intro": "Here is what applies to your question ({n} matching scheme(s)):",
        "benefit": "What you get", "eligibility": "Who qualifies",
        "apply": "Next step", "portal": "Portal",
        "footer": ("_Answered from the local scheme database. Verify current figures "
                   "on the official portal before applying._"),
    },
    "hi": {
        "none": ("आपके सवाल से मेल खाती कोई योजना मेरे रिकॉर्ड में नहीं मिली। कृपया अपने "
                 "ब्लॉक कृषि कार्यालय या नज़दीकी कृषि विज्ञान केंद्र से संपर्क करें — वे आपके "
                 "ज़िले की योजनाओं के बारे में बता सकेंगे।\n\n"
                 "सभी केंद्रीय योजनाएँ आप https://www.myscheme.gov.in पर भी देख सकते हैं।"),
        "intro": "आपके सवाल से जुड़ी जानकारी ({n} योजना मिली):",
        "benefit": "क्या मिलेगा", "eligibility": "कौन पात्र है",
        "apply": "आगे क्या करें", "portal": "पोर्टल",
        "footer": ("_यह उत्तर स्थानीय योजना डेटाबेस से दिया गया है। आवेदन से पहले आधिकारिक "
                   "पोर्टल पर मौजूदा आँकड़े ज़रूर जाँच लें।_"),
    },
}


def _offline_answer(query: str, schemes: list[dict[str, Any]], lang: str = "en") -> str:
    """Structured answer assembled from retrieved schemes, no LLM involved.

    This path carries the whole feature when there is no API key, so it has to
    answer in Hindi too -- otherwise asking in Hindi silently returns English
    and the feature is only half there.
    """
    L = OFFLINE_LABELS.get(lang, OFFLINE_LABELS["en"])
    if not schemes:
        return L["none"]

    lines = [L["intro"].format(n=len(schemes)), ""]
    for s in schemes:
        lines.append(f"**{_field(s, 'name', lang)}**")
        lines.append(f"- {L['benefit']}: {_field(s, 'benefit', lang)}")
        lines.append(f"- {L['eligibility']}: {_field(s, 'eligibility', lang)}")
        lines.append(f"- {L['apply']}: {_field(s, 'how_to_apply', lang)}")
        lines.append(f"- {L['portal']}: {s['portal']}\n")
    lines.append(L["footer"])
    return "\n".join(lines)


def ask(
    query: str,
    context_note: str | None = None,
    history: list[dict[str, str]] | None = None,
    lang: str = "auto",
) -> dict[str, Any]:
    """Answer a farmer's question about government schemes.

    `context_note` carries the current recommendation state (selected state,
    recommended crop) so the assistant can tailor its answer without the farmer
    having to restate it.

    `lang` is "en", "hi", or "auto" to detect from the question's script.
    """
    resolved = detect_language(query) if lang == "auto" else lang
    if resolved not in ("en", "hi"):
        resolved = "en"

    schemes = retrieve(query, k=3)
    sources = [
        {"name": _field(s, "name", resolved), "portal": s["portal"]} for s in schemes
    ]

    if not GROK_KEY:
        return {
            "answer": _offline_answer(query, schemes, resolved),
            "sources": sources,
            "mode": "offline-retrieval",
            "lang": resolved,
            "note": "GROK_API_KEY not set — served from the local scheme database.",
        }

    user_parts = []
    if context_note:
        user_parts.append(f"Farmer's current situation: {context_note}")
    user_parts.append(f"Question: {query}")
    if schemes:
        user_parts.append(
            "Relevant verified scheme information:\n\n" + _context_block(schemes, resolved)
        )
    else:
        user_parts.append(
            "No scheme in the database matched this question. Tell the farmer that "
            "and redirect them to their block agriculture office."
        )

    system = SYSTEM_PROMPT + (HINDI_INSTRUCTION if resolved == "hi" else "")
    messages = [{"role": "system", "content": system}]
    for turn in (history or [])[-6:]:
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        response = httpx.post(
            GROK_URL,
            headers={
                "Authorization": f"Bearer {GROK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROK_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 700,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        # A Hindi question answered in English is a failure the farmer cannot
        # work around, so fall back to the Hindi template rather than ship it.
        if resolved == "hi" and not DEVANAGARI.search(answer):
            return {
                "answer": _offline_answer(query, schemes, "hi"),
                "sources": sources,
                "mode": "offline-fallback",
                "lang": "hi",
                "note": "Model replied in English to a Hindi question; served the Hindi template instead.",
            }
        return {
            "answer": answer,
            "sources": sources,
            "mode": "grok",
            "model": GROK_MODEL,
            "lang": resolved,
        }
    except Exception as exc:  # noqa: BLE001 - degrade rather than fail the request
        return {
            "answer": _offline_answer(query, schemes, resolved),
            "sources": sources,
            "mode": "offline-fallback",
            "lang": resolved,
            "note": f"Grok API unavailable ({type(exc).__name__}); served from the local database.",
        }


def list_schemes() -> list[dict[str, Any]]:
    return [
        {k: v for k, v in s.items() if k not in ("keywords", "keywords_hi")}
        for s in _schemes()
    ]
