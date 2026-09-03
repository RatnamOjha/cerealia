"""
Environment configuration.

The application reads its API key from the environment, but nothing was
populating that environment from `backend/.env` — so a key written to that file
was silently ignored and the app stayed in offline mode with no explanation.
This module loads the file once, at import, before anything reads a setting.

Deliberately dependency-free: this is fifteen lines and one less package to
install on a demo machine.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_PATH) -> list[str]:
    """Load KEY=VALUE lines into os.environ. Returns the names it set.

    Real environment variables always win, so `GROK_API_KEY=... uvicorn ...`
    overrides the file rather than being overridden by it.
    """
    if not path.exists():
        return []

    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


LOADED_KEYS = load_env()


def get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def redact(value: str) -> str:
    """Safe-to-log fingerprint of a secret — never the secret itself."""
    if not value:
        return "not set"
    return f"set ({len(value)} chars, ends …{value[-4:]})"
