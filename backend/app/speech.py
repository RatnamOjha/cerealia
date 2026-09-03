"""
Speech-to-text for Hindi and English.

Supports two providers, chosen automatically from the API key prefix:
Groq (`gsk_...`, Whisper, free tier) and xAI (`xai-...`, billed per hour).
They differ in the multipart fields they expect, which is the only reason
this needs a branch at all.

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

from . import config

STT_TIMEOUT = float(os.getenv("STT_TIMEOUT") or os.getenv("GROK_STT_TIMEOUT") or "45")

# Roughly 60 seconds of Opus at typical MediaRecorder bitrates. A farmer asking
# a question does not need more, and a cap keeps a runaway recording from
# burning API credit.
MAX_AUDIO_BYTES = 8 * 1024 * 1024

SUPPORTED_LANGUAGES = {"hi", "en"}


def _key() -> str:
    return config.api_key()


def available() -> bool:
    return bool(_key())


def describe() -> dict[str, Any]:
    """What the interface should say about speech, without exposing the key."""
    cfg = config.provider_config()
    return {
        "provider": cfg["provider"] if available() else "browser",
        "label": cfg["label"] if available() else "Browser speech recognition",
        "model": cfg["stt_model"] if available() else None,
        "server_side": available(),
        "languages": sorted(SUPPORTED_LANGUAGES),
    }


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
    cfg = config.provider_config()

    # Groq follows OpenAI's transcription shape (model + response_format);
    # xAI's endpoint is itself the model and takes `format` for normalisation.
    if cfg["stt_style"] == "openai":
        form = {"model": cfg["stt_model"], "language": lang, "response_format": "json"}
    else:
        form = {"language": lang, "format": "true"}

    try:
        response = httpx.post(
            cfg["stt_url"],
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename, audio, content_type)},
            # httpx wants a mapping. Provider docs show a list of tuples, which
            # is valid for `requests` but makes httpx fail while building the
            # multipart body -- the request then goes out with no file attached
            # and the transcript comes back empty.
            data=form,
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
        "provider": cfg["provider"],
        "model": cfg["stt_model"],
        "audio_bytes": len(audio),
    }
