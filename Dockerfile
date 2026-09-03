# Cerealia — single-container deployment.
#
# The API also serves the built frontend, so this is one free service rather
# than two: no CORS to configure, no second host to keep awake, and the browser
# talks to the same origin it loaded from.
#
# Targets Hugging Face Spaces (Docker SDK), which is free with no card and no
# cold-start penalty. The same image runs anywhere that accepts a Dockerfile.

# --- stage 1: build the frontend -------------------------------------------
FROM node:20-slim AS web

WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent

COPY frontend/ ./
RUN npm run build


# --- stage 2: the service ---------------------------------------------------
FROM python:3.12-slim

# Spaces run as a non-root user; matching that locally avoids permission
# surprises when the same image is run elsewhere.
RUN useradd -m -u 1000 app

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=app:app backend/ ./backend/
COPY --chown=app:app --from=web /web/dist ./frontend/dist

# Train at build time. It takes about a second and keeps a 5 MB binary out of
# version control, so the deployed model is always the one this code produces.
RUN cd backend && python train.py && chown -R app:app models

USER app

# Spaces routes to 7860; PORT is honoured for other hosts.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"

CMD ["sh", "-c", "cd backend && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
