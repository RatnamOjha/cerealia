#!/usr/bin/env bash
#
# Deploy Cerealia to Hugging Face Spaces — free, no card, no cold starts.
#
#   ./deploy.sh <your-hf-username>
#
# Run `hf auth login` once first. The API key is uploaded as a Space secret,
# never committed.

set -euo pipefail

USER_NAME="${1:-}"
SPACE_NAME="${SPACE_NAME:-cerealia}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
dim()   { printf '\033[0;90m%s\033[0m\n' "$1"; }
die()   { printf '\033[0;31m%s\033[0m\n' "$1" >&2; exit 1; }

[ -n "$USER_NAME" ] || die "Usage: ./deploy.sh <your-hf-username>"
command -v hf >/dev/null || die "hf CLI not found. Install: curl -LsSf https://hf.co/cli/install.sh | bash"
hf auth whoami >/dev/null 2>&1 || die "Not logged in. Run: hf auth login"

REPO="$USER_NAME/$SPACE_NAME"

green "Creating Space $REPO (Docker SDK)…"
hf repos create "$REPO" --type space --sdk docker --exist-ok

# The Space README carries the Docker config Spaces reads on build.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cat > "$STAGE/README.md" <<'CARD'
---
title: Cerealia
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI crop intelligence - soil, climate, price and risk in one answer
---

# Cerealia

AI crop intelligence for farmers. Pick a region on the globe and get crops
ranked by risk-adjusted expected return, with a Hindi voice assistant that
answers both "what should I grow?" and "what help can I get?".

Built on 246,091 official Government of India crop production records.
Source: https://github.com/RatnamOjha/cerealia
CARD

green "Uploading source…"
for item in Dockerfile .dockerignore backend frontend; do
  [ -e "$ROOT/$item" ] || die "missing $item"
done

hf upload "$REPO" "$ROOT" . \
  --type space \
  --commit-message "Deploy Cerealia" \
  --exclude "**/.venv/**" "**/node_modules/**" "**/dist/**" "**/__pycache__/**" \
            ".git/**" ".env" "docs/screenshots/**" "docs/*.pptx" \
            "backend/models/**" "*.log" "**/.DS_Store"

hf upload "$REPO" "$STAGE/README.md" README.md --type space \
  --commit-message "Space card"

# Ship the key as a secret if one is configured locally.
KEY="$(grep -hoE '^[[:space:]]*(GROK|GROQ|XAI)_API_KEY[[:space:]]*=.*' "$ROOT/.env" "$ROOT/backend/.env" 2>/dev/null \
       | head -1 | sed -E "s/^[^=]*=[[:space:]]*//; s/^['\"]//; s/['\"]$//" || true)"
if [ -n "$KEY" ]; then
  green "Uploading API key as a Space secret…"
  hf spaces secrets add "$REPO" --secrets "GROK_API_KEY=$KEY" >/dev/null
  dim   "  stored as a secret, not committed"
else
  dim "No API key found locally. The Space will run in offline mode."
  dim "Add one later: hf spaces secrets add $REPO --secrets GROK_API_KEY=..."
fi

green "Building…"
hf spaces wait "$REPO" --timeout 20m || dim "Still building — check the logs."

green ""
green "  Live at  https://huggingface.co/spaces/$REPO"
dim   "  Logs:    hf spaces logs $REPO --follow"
