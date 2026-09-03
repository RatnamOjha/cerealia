# Deploying Cerealia

One container serves both the API and the web app, so this is a single service.
No CORS to configure, no second host to keep awake.

## Why Render

The obvious candidates mostly do not work for this app, and it is worth
recording why so nobody re-litigates it:

| | Runs the Python API? | Card required | Cold start |
|---|---|---|---|
| **Render (Hobby)** | yes, Docker | no | 30-60 s after 15 min idle |
| Hugging Face Spaces | yes, but Docker Spaces now **require PRO** | no (PRO is paid) | — |
| Netlify / GitHub Pages | **no** — static + JS/TS/Go functions only | no | — |
| Google Cloud Run | yes | yes, and an RBI e-mandate in India | seconds |
| Fly.io / Railway / Koyeb | free tiers ended or closed to new users | yes | — |

The static hosts are the trap: this app is half backend. Without the API there
are no recommendations, no chatbot and no speech — just a map that does nothing.

## Deploy

1. Push the repo to GitHub.
2. Sign in at [render.com](https://render.com) with GitHub — no card.
3. **New → Blueprint**, pick this repo. Render reads `render.yaml`.
4. It prompts for `GROK_API_KEY` (marked `sync: false`, so it is never in the
   repo). Paste it, or skip — see below.
5. First build takes 5-10 minutes: it builds the frontend, installs
   scikit-learn and pandas, and trains the model.

Render redeploys automatically on every push to `main` — no CI workflow needed.

## Custom domain

Render issues and renews TLS certificates for custom domains automatically, on
the free plan.

1. Render dashboard → your service → **Settings → Custom Domains → Add**.
2. It shows the exact DNS records to create.
3. At your registrar (Porkbun: **Details → DNS Records**), add them — a `CNAME`
   for `www`, and Porkbun's `ALIAS` at the apex, pointing at the value Render
   gives you.
4. **Delete any `AAAA` records.** Render is IPv4-only and stray `AAAA` records
   cause intermittent failures that are miserable to debug.

DNS takes minutes to a few hours. The certificate is issued once it resolves.

## The sleep tradeoff

Free instances spin down after 15 minutes idle; the next visitor waits 30-60 s.

`.github/workflows/keep-awake.yml` pings `/api/health` every 10 minutes to keep
the timer from expiring — set the `APP_URL` repository variable to enable it.
GitHub's scheduler is loose, so this narrows cold starts rather than removing
them. A paid instance is the only way to remove them entirely.

## Without an API key

The app still runs. The chatbot falls back to local retrieval over the scheme
database and speech falls back to the browser recogniser — both answer in
Hindi. The model, the recommendation engine and all the data are local to the
container, so everything else is unaffected.

## Hugging Face

`deploy.sh` still works if you ever subscribe to PRO — Docker Spaces returned
`402 Payment Required` on the free tier as of September 2026.
