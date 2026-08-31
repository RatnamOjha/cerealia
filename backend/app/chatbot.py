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

DATA_DIR = Path(__file__).resolve().parent / "data"

GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")
GROK_KEY = os.getenv("GROK_API_KEY", "").strip()
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
5. If the farmer writes in Hindi or another Indian language, reply in that
   language.
6. Keep answers under 200 words unless the farmer asks for detail."""


@lru_cache(maxsize=1)
def _schemes() -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / "schemes.json").read_text())["schemes"]


def _tokenise(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Score every scheme against the query and return the best k.

    A keyword overlap score is enough at this corpus size (12 schemes) and keeps
    the whole thing dependency-free and instant. Swapping in sentence-embedding
    retrieval becomes worthwhile once the corpus reaches hundreds of documents.
    """
    q = _tokenise(query)
    if not q:
        return []

    scored = []
    for scheme in _schemes():
        keyword_hits = sum(
            2.0 for kw in scheme["keywords"] if kw in query.lower()
        )
        name_hits = len(q & _tokenise(scheme["name"])) * 1.5
        body = " ".join([scheme["benefit"], scheme["eligibility"], scheme["category"]])
        body_hits = len(q & _tokenise(body)) * 0.4
        total = keyword_hits + name_hits + body_hits
        if total > 0:
            scored.append((total, scheme))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [s for _, s in scored[:k]]


def _context_block(schemes: list[dict[str, Any]]) -> str:
    parts = []
    for s in schemes:
        parts.append(
            f"SCHEME: {s['name']}\n"
            f"Category: {s['category']}\n"
            f"Benefit: {s['benefit']}\n"
            f"Eligibility: {s['eligibility']}\n"
            f"How to apply: {s['how_to_apply']}\n"
            f"Official portal: {s['portal']}"
        )
    return "\n\n---\n\n".join(parts)


def _offline_answer(query: str, schemes: list[dict[str, Any]]) -> str:
    """Structured answer assembled from retrieved schemes, no LLM involved."""
    if not schemes:
        return (
            "I could not match your question to a scheme in my records. "
            "Please contact your block agriculture office or the nearest Krishi "
            "Vigyan Kendra — they can advise on schemes specific to your district.\n\n"
            "You can also browse every central scheme at https://www.myscheme.gov.in"
        )

    lines = [f"Here is what applies to your question ({len(schemes)} matching scheme(s)):\n"]
    for s in schemes:
        lines.append(f"**{s['name']}**")
        lines.append(f"- What you get: {s['benefit']}")
        lines.append(f"- Who qualifies: {s['eligibility']}")
        lines.append(f"- Next step: {s['how_to_apply']}")
        lines.append(f"- Portal: {s['portal']}\n")
    lines.append(
        "_Answered from the local scheme database. Verify current figures on the "
        "official portal before applying._"
    )
    return "\n".join(lines)


def ask(
    query: str,
    context_note: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Answer a farmer's question about government schemes.

    `context_note` carries the current recommendation state (selected state,
    recommended crop) so the assistant can tailor its answer without the farmer
    having to restate it.
    """
    schemes = retrieve(query, k=3)

    if not GROK_KEY:
        return {
            "answer": _offline_answer(query, schemes),
            "sources": [{"name": s["name"], "portal": s["portal"]} for s in schemes],
            "mode": "offline-retrieval",
            "note": "GROK_API_KEY not set — served from the local scheme database.",
        }

    user_parts = []
    if context_note:
        user_parts.append(f"Farmer's current situation: {context_note}")
    user_parts.append(f"Question: {query}")
    if schemes:
        user_parts.append(
            "Relevant verified scheme information:\n\n" + _context_block(schemes)
        )
    else:
        user_parts.append(
            "No scheme in the database matched this question. Tell the farmer that "
            "and redirect them to their block agriculture office."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        return {
            "answer": answer,
            "sources": [{"name": s["name"], "portal": s["portal"]} for s in schemes],
            "mode": "grok",
            "model": GROK_MODEL,
        }
    except Exception as exc:  # noqa: BLE001 - degrade rather than fail the request
        return {
            "answer": _offline_answer(query, schemes),
            "sources": [{"name": s["name"], "portal": s["portal"]} for s in schemes],
            "mode": "offline-fallback",
            "note": f"Grok API unavailable ({type(exc).__name__}); served from the local database.",
        }


def list_schemes() -> list[dict[str, Any]]:
    return [
        {k: v for k, v in s.items() if k != "keywords"}
        for s in _schemes()
    ]
