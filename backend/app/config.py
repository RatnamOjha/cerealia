"""
Environment configuration and AI provider detection.

Two things were silently wrong before this existed:

  1. Nothing loaded a `.env` file at all, so a key written to one was ignored
     and the app stayed in offline mode with no explanation.
  2. "Grok" (xAI) and "Groq" (groq.com) are different companies with nearly
     identical names, and a key from one will never work against the other's
     endpoint. Rather than make the user care, the provider is detected from
     the key's own prefix.

Deliberately dependency-free: this is short enough not to justify a package on
a demo machine.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent

# Both are checked because either is a reasonable place to put it, and being
# wrong about the location should not cost anyone an afternoon.
ENV_CANDIDATES = [REPO_ROOT / ".env", BACKEND_DIR / ".env"]


def parse_env(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines, tolerating the shapes people actually write.

    Accepts `KEY=value`, `KEY = value`, quoted values, and a leading `export`.
    Whitespace around the `=` is the common case and silently breaking on it
    is a nasty way to lose an hour.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env(paths: list[Path] | None = None) -> list[str]:
    """Load .env files into os.environ. Returns the names it set.

    Real environment variables always win, so `GROK_API_KEY=... uvicorn ...`
    overrides the file rather than the other way round.
    """
    loaded = []
    for path in paths or ENV_CANDIDATES:
        if not path.exists():
            continue
        for key, value in parse_env(path.read_text(encoding="utf-8")).items():
            if value and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


LOADED_KEYS = load_env()


# --- provider detection ----------------------------------------------------

# Providers retire models without warning -- llama-3.3-70b-versatile was the
# documented default and simply stopped existing, which surfaced as an opaque
# HTTP error mid-demo. So a preference list is resolved against what the key
# can actually see, rather than one hard-coded name.
CHAT_PREFERENCES = {
    "groq": [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "groq/compound",
    ],
    "xai": ["grok-4", "grok-3", "grok-2-latest"],
}

PROVIDERS = {
    "groq": {
        "label": "Groq",
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "chat_model": "openai/gpt-oss-120b",
        "models_url": "https://api.groq.com/openai/v1/models",
        "stt_url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "stt_model": "whisper-large-v3-turbo",
        "stt_style": "openai",     # sends `model` + `response_format`
        "note": "Whisper STT is free on Groq's tier: 2,000 transcriptions/day.",
    },
    "xai": {
        "label": "xAI (Grok)",
        "chat_url": "https://api.x.ai/v1/chat/completions",
        "chat_model": "grok-3",
        "stt_url": "https://api.x.ai/v1/stt",
        "stt_model": None,          # the endpoint is the model
        "stt_style": "xai",         # sends `format` instead of `response_format`
        "models_url": "https://api.x.ai/v1/models",
        "note": "xAI STT is billed per hour of audio.",
    },
}

_resolved_model: dict[str, str] = {}


def resolve_chat_model(provider: str, key: str, fallback: str) -> str:
    """First preferred model the key can actually reach.

    Asks the provider once and caches. Any failure falls back to the static
    default -- a slow or unreachable models endpoint must not stop the app
    from starting.
    """
    if provider in _resolved_model:
        return _resolved_model[provider]

    prefs = CHAT_PREFERENCES.get(provider, [])
    url = PROVIDERS.get(provider, {}).get("models_url")
    chosen = fallback

    if key and url and prefs:
        try:
            import httpx

            resp = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=8)
            if resp.status_code == 200:
                available = {m.get("id") for m in resp.json().get("data", [])}
                chosen = next((m for m in prefs if m in available), fallback)
        except Exception:  # noqa: BLE001 - never block startup on this
            pass

    _resolved_model[provider] = chosen
    return chosen


def api_key() -> str:
    """The configured key, whichever of the accepted names it was written under."""
    for name in ("GROK_API_KEY", "GROQ_API_KEY", "XAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def detect_provider(key: str = "") -> str:
    """Identify the provider from the key prefix.

    Groq issues `gsk_...`, xAI issues `xai-...`. An explicit AI_PROVIDER
    overrides the guess for anything unusual.
    """
    forced = os.getenv("AI_PROVIDER", "").strip().lower()
    if forced in PROVIDERS:
        return forced

    key = key or api_key()
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("xai-"):
        return "xai"
    # Unrecognised but present: Groq is the safer default, being the free one.
    return "groq" if key else "none"


def provider_config() -> dict:
    """Resolved endpoints and models for the active provider, with overrides."""
    name = detect_provider()
    base = PROVIDERS.get(name, PROVIDERS["groq"])
    key = api_key()
    pinned = os.getenv("CHAT_MODEL") or os.getenv("GROK_MODEL")
    model = pinned or resolve_chat_model(name, key, base["chat_model"])
    return {
        "provider": name,
        "label": base["label"],
        "chat_url": os.getenv("CHAT_API_URL") or os.getenv("GROK_API_URL") or base["chat_url"],
        "chat_model": model,
        "stt_url": os.getenv("STT_API_URL") or os.getenv("GROK_STT_URL") or base["stt_url"],
        "stt_model": os.getenv("STT_MODEL") or base["stt_model"],
        "stt_style": base["stt_style"],
        "note": base["note"],
    }


def redact(value: str) -> str:
    """Safe-to-log fingerprint of a secret — never the secret itself."""
    if not value:
        return "not set"
    return f"set ({len(value)} chars, starts {value[:4]}…)"
