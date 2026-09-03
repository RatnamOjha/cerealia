"""
Speech-to-text for Hindi and English, via the xAI (Grok) STT API.

Why this runs server-side rather than in the browser:

  1. The API key never reaches the client. A key shipped to the browser is a
     key published to the world.
  2. It works in every browser. The Web Speech API is Chrome and Edge only —
     Safari and Firefox farmers get nothing. MediaRecorder, which is what we
     use to capture the audio, is universal.
  3. Hindi accuracy. A model trained for Hindi beats the browser's generic
     recogniser on rural vocabulary and accents.

The browser recogniser is kept as an automatic fallback when no key is
configured, so the feature degrades instead of disappearing.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from . import config  # noqa: F401  -- loads backend/.env

STT_URL = os.getenv("GROK_STT_URL", "https://api.x.ai/v1/stt")
STT_TIMEOUT = float(os.getenv("GROK_STT_TIMEOUT", "45"))

# Roughly 60 seconds of Opus at typical MediaRecorder bitrates. A farmer asking
# a question does not need more, and a cap keeps a runaway recording from
# burning API credit.
MAX_AUDIO_BYTES = 8 * 1024 * 1024

SUPPORTED_LANGUAGES = {"hi", "en"}


def _key() -> str:
    return (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()


def available() -> bool:
    return bool(_key())


def transcribe(
    audio: bytes,
    filename: str = "speech.webm",
    content_type: str = "audio/webm",
    language: str = "hi",
) -> dict[str, Any]:
    """Transcribe recorded audio. Never raises — returns an `ok` flag.

    The caller is a farmer holding a microphone button, so every failure has to
    come back as something the interface can say out loud, not a stack trace.
    """
    if not audio:
        return {"ok": False, "error": "empty", "detail": "No audio was received."}

    if len(audio) > MAX_AUDIO_BYTES:
        return {
            "ok": False,
            "error": "too_large",
            "detail": f"Recording is {len(audio) // 1024} KB; the limit is "
                      f"{MAX_AUDIO_BYTES // 1024} KB. Please ask a shorter question.",
        }

    key = _key()
    if not key:
        return {
            "ok": False,
            "error": "no_key",
            "detail": "No xAI API key configured. Falling back to browser speech recognition.",
        }

    lang = language if language in SUPPORTED_LANGUAGES else "hi"

    try:
        response = httpx.post(
            STT_URL,
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename, audio, content_type)},
            # httpx wants a mapping here. xAI's own example passes a list of
            # tuples, which is valid for `requests` but makes httpx fail while
            # building the multipart body -- and the request goes out with no
            # file attached at all.
            # Inverse text normalisation matters: a farmer saying "छह हज़ार
            # रुपये" should come back as digits, because the scheme text it has
            # to match is written in digits.
            data={"language": lang, "format": "true"},
            timeout=STT_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        body = ""
        try:
            body = exc.response.text[:200]
        except Exception:  # noqa: BLE001
            pass
        return {
            "ok": False,
            "error": f"http_{exc.response.status_code}",
            "detail": f"Speech service returned {exc.response.status_code}. {body}".strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]}

    # The response shape is documented as {"text": ...}, but be tolerant of the
    # transcript arriving under a different key rather than returning silence.
    text = (
        payload.get("text")
        or payload.get("transcript")
        or (payload.get("results") or {}).get("text")
        or ""
    ).strip()

    if not text:
        return {
            "ok": False,
            "error": "empty_transcript",
            "detail": "Nothing was recognised. Please speak a little closer to the microphone.",
        }

    return {
        "ok": True,
        "text": text,
        "language": lang,
        "provider": "xai-stt",
        "audio_bytes": len(audio),
    }
