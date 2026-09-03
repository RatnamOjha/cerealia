"""
Farm advisory chatbot.

Answers two kinds of question, because a farmer does not separate them:

  1. "What should I grow here?" -- answered from the recommendation engine,
     with the real ranked crops, their margins and their water needs.
  2. "What help can I get?" -- answered from the curated scheme database.

The first was originally missing, which meant the single most natural question
a farmer could ask was the one the assistant refused, even though the rest of
the application had just computed the answer.

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

from . import config
from .recommender import get_state, recommend_for_state

DATA_DIR = Path(__file__).resolve().parent / "data"

# Endpoint and model follow whichever provider the key belongs to. Both expose
# an OpenAI-compatible chat completions API, so only the URL and model differ.
_CFG = config.provider_config()
GROK_URL = _CFG["chat_url"]
GROK_MODEL = _CFG["chat_model"]
GROK_KEY = config.api_key()
PROVIDER = _CFG["provider"]
PROVIDER_LABEL = _CFG["label"]
REQUEST_TIMEOUT = float(os.getenv("CHAT_TIMEOUT") or os.getenv("GROK_TIMEOUT") or "25")

SYSTEM_PROMPT = """You are Cerealia, an agricultural advisor for Indian farmers.

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

CROP_SYSTEM = """

You are now answering a question about WHICH CROP TO GROW, not about a scheme.
The ranked crops below were computed by this system's own recommendation engine
from soil, climate, measured government yield data, price, water need and risk.

- Use those crops and those exact figures. Do not invent crops or numbers.
- Lead with the top recommendation and say plainly why it suits this land.
- Mention the money (net return per hectare per year) and the irrigation need,
  because those decide whether the farmer can actually take it on.
- If the top crop has low climate-fit confidence, say so honestly.
- Do not discuss schemes unless the farmer also asked about them."""

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

# Phrases that mean "tell me what to plant" rather than "tell me about a
# scheme". Substring matching, because Devanagari inflects with suffixes and
# English phrasing varies far more than a keyword list can enumerate.
CROP_INTENT = [
    # English
    "which crop", "what crop", "which crops", "what crops",
    "should i grow", "should i plant", "should i sow", "can i grow",
    "what to grow", "what to plant", "best crop", "recommend a crop",
    "crop recommendation", "suitable crop", "grow here", "plant this season",
    "most profitable crop", "which seed",
    # Hindi
    "कौन सी फसल", "कौनसी फसल", "कौन सा फसल", "कौन-सी फसल",
    "क्या उगा", "क्या लगा", "क्या बो", "क्या खेती",
    "फसल उगा", "फसल लगा", "फसल बो", "फसल की सलाह", "फसल चुन",
    "बुवाई", "कौन सी खेती", "सबसे अच्छी फसल", "ज्यादा मुनाफ",
    "कौन सी फ़सल", "क्या फसल",
]

CROP_LABELS = {
    "en": {
        "intro": "For {state}, these are the best-scoring crops right now:",
        "fit": "climate fit", "net": "net", "per_year": "/ha per year",
        "water": "water", "msp": "MSP-backed",
        "footer": ("Ranked on risk-adjusted expected return — climate fitness, "
                   "measured yield, price, water need and risk combined. "
                   "Open a crop card on the left for the full breakdown."),
        "no_state": ("Tap a state on the map first, then ask again and I will "
                     "rank the crops for that region."),
    },
    "hi": {
        "intro": "{state} के लिए इस समय सबसे उपयुक्त फसलें ये हैं:",
        "fit": "जलवायु अनुकूलता", "net": "शुद्ध लाभ", "per_year": "/हेक्टेयर प्रति वर्ष",
        "water": "पानी", "msp": "एमएसपी समर्थित",
        "footer": ("यह क्रम जोखिम-समायोजित अपेक्षित आय पर आधारित है — जलवायु अनुकूलता, "
                   "मापी गई पैदावार, भाव, पानी की ज़रूरत और जोखिम, सब मिलाकर। "
                   "पूरा ब्यौरा बाईं ओर फसल कार्ड खोलकर देखें।"),
        "no_state": ("पहले नक्शे पर अपना राज्य चुनिए, फिर दोबारा पूछिए — मैं उस "
                     "क्षेत्र की फसलें क्रम से बताऊँगा।"),
    },
}

WATER_HI = {
    "rainfed": "बारिश पर निर्भर", "light": "थोड़ी सिंचाई",
    "moderate": "मध्यम सिंचाई", "heavy": "अधिक सिंचाई",
}


def is_crop_question(query: str) -> bool:
    """True when the farmer is asking what to plant, not about a scheme."""
    lowered = query.lower()
    return any(k in lowered or k in query for k in CROP_INTENT)


def _rupees(amount: float, lang: str) -> str:
    if amount >= 100000:
        return f"₹{amount / 100000:.2f} लाख" if lang == "hi" else f"₹{amount / 100000:.2f} lakh"
    return f"₹{amount:,.0f}"


def crop_context(state_id: str, lang: str, top_n: int = 5) -> tuple[str, list[dict]]:
    """Ranked crops for a state, as grounding text plus structured sources."""
    result = recommend_for_state(state_id, top_n=top_n)
    state = result["state"]
    lines = [
        f"State: {state['name']} ({state['zone']}, {state['soil']} soil)",
        f"Soil: N {result['site']['N']}, P {result['site']['P']}, K {result['site']['K']}, "
        f"pH {result['site']['ph']}; {result['rainfall_annual_mm']} mm annual rainfall",
        "",
        "RANKED CROPS (already computed by the recommendation engine — use these "
        "exact figures, do not invent others):",
    ]
    sources = []
    for i, rec in enumerate(result["recommendations"], 1):
        econ = rec["economics"]
        name = rec["display"]
        if lang == "hi":
            name = f"{rec.get('display_hi') or rec['display']}"
        lines.append(
            f"{i}. {rec['display']} — climate fit {rec['agro_fit_pct']}% ({rec['confidence']}), "
            f"net ₹{econ['net_profit_per_ha_year']:,.0f}/ha/year, "
            f"yield {econ['yield_t_ha_used']} t/ha ({econ['yield_source']}), "
            f"water: {rec['water']['label']}"
            + (", MSP-backed" if rec["msp_backed"] else "")
        )
        sources.append({"name": name, "portal": ""})
    lines.append("")
    lines.append(
        "Explain the top two or three in plain language: why they suit this soil "
        "and climate, what they earn, and what irrigation they need."
    )
    return "\n".join(lines), sources


def _offline_crop_answer(state_id: str, lang: str, top_n: int = 5) -> tuple[str, list[dict]]:
    """Ranked crops rendered directly, with no model involved."""
    L = CROP_LABELS.get(lang, CROP_LABELS["en"])
    result = recommend_for_state(state_id, top_n=top_n)
    state_name = result["state"]["name"]

    out = [L["intro"].format(state=state_name), ""]
    sources = []
    for i, rec in enumerate(result["recommendations"], 1):
        econ = rec["economics"]
        name = rec.get("display_hi") if lang == "hi" else None
        name = name or rec["display"]
        water = rec["water"]["verdict"]
        water_txt = WATER_HI.get(water, water) if lang == "hi" else rec["water"]["label"]
        tag = f" · {L['msp']}" if rec["msp_backed"] else ""
        out.append(f"**{i}. {name}**{tag}")
        out.append(
            f"- {L['fit']}: {rec['agro_fit_pct']}% ({rec['confidence']})"
        )
        out.append(
            f"- {L['net']}: {_rupees(econ['net_profit_per_ha_year'], lang)}{L['per_year']}"
        )
        out.append(f"- {L['water']}: {water_txt}\n")
        sources.append({"name": name, "portal": ""})
    out.append(f"_{L['footer']}_")
    return "\n".join(out), sources


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
    state_id: str | None = None,
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

    # --- crop questions go to the recommendation engine, not the schemes ----
    crop_mode = is_crop_question(query)
    if crop_mode:
        if not (state_id and get_state(state_id)):
            L = CROP_LABELS.get(resolved, CROP_LABELS["en"])
            return {
                "answer": L["no_state"],
                "sources": [],
                "mode": "needs-state",
                "topic": "crops",
                "lang": resolved,
            }
        try:
            grounding, sources = crop_context(state_id, resolved)
        except (KeyError, FileNotFoundError) as exc:
            return {
                "answer": str(exc),
                "sources": [],
                "mode": "error",
                "topic": "crops",
                "lang": resolved,
            }
        schemes = []
    else:
        schemes = retrieve(query, k=3)
        sources = [
            {"name": _field(s, "name", resolved), "portal": s["portal"]} for s in schemes
        ]
        grounding = None

    if not GROK_KEY:
        if crop_mode:
            answer, sources = _offline_crop_answer(state_id, resolved)
        else:
            answer = _offline_answer(query, schemes, resolved)
        return {
            "answer": answer,
            "sources": sources,
            "mode": "offline-retrieval",
            "topic": "crops" if crop_mode else "schemes",
            "lang": resolved,
            "note": "No API key set — served from local data.",
        }

    user_parts = []
    if context_note:
        user_parts.append(f"Farmer's current situation: {context_note}")
    user_parts.append(f"Question: {query}")
    if crop_mode:
        user_parts.append(grounding)
    elif schemes:
        user_parts.append(
            "Relevant verified scheme information:\n\n" + _context_block(schemes, resolved)
        )
    else:
        user_parts.append(
            "No scheme in the database matched this question. Tell the farmer that "
            "and redirect them to their block agriculture office."
        )

    system_extra = CROP_SYSTEM if crop_mode else ""

    system = SYSTEM_PROMPT + system_extra + (HINDI_INSTRUCTION if resolved == "hi" else "")
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
            hi_fallback, hi_sources = (
                _offline_crop_answer(state_id, "hi") if crop_mode
                else (_offline_answer(query, schemes, "hi"), sources)
            )
            return {
                "answer": hi_fallback,
                "sources": hi_sources,
                "mode": "offline-fallback",
                "lang": "hi",
                "note": "Model replied in English to a Hindi question; served the Hindi template instead.",
            }
        return {
            "answer": answer,
            "sources": sources,
            "mode": "llm",
            "provider": PROVIDER_LABEL,
            "model": GROK_MODEL,
            "topic": "crops" if crop_mode else "schemes",
            "lang": resolved,
        }
    except Exception as exc:  # noqa: BLE001 - degrade rather than fail the request
        if crop_mode:
            fallback, sources = _offline_crop_answer(state_id, resolved)
        else:
            fallback = _offline_answer(query, schemes, resolved)
        return {
            "answer": fallback,
            "sources": sources,
            "mode": "offline-fallback",
            "topic": "crops" if crop_mode else "schemes",
            "lang": resolved,
            "note": f"{PROVIDER_LABEL} unavailable ({type(exc).__name__}); served from the local database.",
        }


def list_schemes() -> list[dict[str, Any]]:
    return [
        {k: v for k, v in s.items() if k not in ("keywords", "keywords_hi")}
        for s in _schemes()
    ]
