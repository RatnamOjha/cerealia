#!/usr/bin/env bash
#
# Start Cerealia — backend and frontend together, one command.
#
#   ./dev.sh
#
# Ctrl-C stops both. Everything below is idempotent: it creates the venv,
# installs dependencies and trains the model only when they are missing, so
# this is also a safe first-run command on a fresh clone.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
PY="$BACKEND/.venv/bin/python"
API_PORT="${API_PORT:-8010}"
WEB_PORT="${WEB_PORT:-5173}"

green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
dim()   { printf '\033[0;90m%s\033[0m\n' "$1"; }
warn()  { printf '\033[0;33m%s\033[0m\n' "$1"; }
die()   { printf '\033[0;31m%s\033[0m\n' "$1" >&2; exit 1; }

cleanup() {
  dim ""
  dim "Stopping…"
  # Kill the whole process group of each child so uvicorn's reloader and
  # vite's esbuild helper go down too, rather than being orphaned on the port.
  for pid in "${API_PID:-}" "${WEB_PID:-}"; do
    [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  green "Stopped."
}
trap cleanup EXIT INT TERM

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# --- preflight -------------------------------------------------------------

[ -d "$BACKEND" ] || die "Run this from the repository root."

if [ ! -x "$PY" ]; then
  warn "Creating Python virtual environment…"
  python3 -m venv "$BACKEND/.venv" || die "Could not create venv. Is python3 installed?"
  "$PY" -m pip install -q --upgrade pip
fi

if ! "$PY" -c "import fastapi, sklearn, httpx" >/dev/null 2>&1; then
  warn "Installing Python dependencies…"
  "$PY" -m pip install -q -r "$BACKEND/requirements.txt"
fi

if [ ! -f "$BACKEND/models/crop_suitability.joblib" ]; then
  warn "Training the model (about one second)…"
  ( cd "$BACKEND" && "$PY" train.py >/dev/null ) || die "Training failed."
fi

if [ ! -d "$FRONTEND/node_modules" ]; then
  warn "Installing frontend dependencies…"
  ( cd "$FRONTEND" && npm install --silent ) || die "npm install failed."
fi

for port in "$API_PORT" "$WEB_PORT"; do
  if port_busy "$port"; then
    die "Port $port is already in use. Free it, or run: API_PORT=8011 WEB_PORT=5174 ./dev.sh"
  fi
done

# --- start -----------------------------------------------------------------

green "Cerealia"
dim   "─────────────────────────────────────────"

( cd "$BACKEND" && exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" --reload ) &
API_PID=$!

# Wait for the API before starting Vite, so the first page load already has a
# live backend and does not flash the "API offline" state.
if ! curl -sf --retry 60 --retry-delay 1 --retry-connrefused --retry-all-errors \
      -o /tmp/cerealia-health.json "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then
  die "Backend did not come up on port $API_PORT."
fi

"$PY" - "$API_PORT" <<'PYEOF'
import json, sys
h = json.load(open("/tmp/cerealia-health.json"))
p, s = h.get("provider", {}), h.get("stt", {})
ok = "\033[0;32m✓\033[0m"
dot = "\033[0;33m○\033[0m"
print(f"  {ok} API        http://localhost:{sys.argv[1]}")
print(f"  {ok} Model      {h['metrics']['cv_accuracy_mean']*100:.1f}% CV" if h.get("metrics") else "")
if h.get("chatbot_mode") == "llm":
    print(f"  {ok} Chatbot    {p.get('label')} · {p.get('chat_model')}")
else:
    print(f"  {dot} Chatbot    local scheme database (no API key)")
if s.get("server_side"):
    print(f"  {ok} Speech     {s.get('label')} · {s.get('model')}")
else:
    print(f"  {dot} Speech     browser recognition (Chrome/Edge only)")
PYEOF

( cd "$FRONTEND" && exec npm run dev -- --port "$WEB_PORT" --strictPort ) &
WEB_PID=$!

dim   "─────────────────────────────────────────"
green "  Open  http://localhost:$WEB_PORT"
dim   "  Ctrl-C to stop both"
dim   ""

wait
