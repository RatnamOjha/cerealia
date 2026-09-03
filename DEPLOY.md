# Deploying Cerealia — zero cost

One container serves both the API and the web app, so this is a single free
service. No CORS to configure, no second host to keep awake.

## Why Hugging Face Spaces

| | Free tier | Cold start | Card required |
|---|---|---|---|
| **Hugging Face Spaces** | 2 vCPU, 16 GB, unlimited | none while awake | no |
| Render free | 512 MB | ~50 s spin-up after 15 min idle | no |
| Fly.io / Railway | trial credits only | — | yes |
| Vercel / Netlify | frontend only | — | no |

Render's cold start is the problem: a reviewer clicking your link waits almost a
minute staring at nothing. Spaces also suits an ML project — scikit-learn and
pandas are already expected there.

Vercel would work for the frontend, but then the backend still needs a home and
you are maintaining two deployments plus CORS.

## Deploy

```bash
# once
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login          # paste a token from huggingface.co/settings/tokens

# every time
./deploy.sh <your-hf-username>
```

The script creates the Space, uploads the source, ships your API key as a Space
**secret** (never committed), and waits for the build. It prints the live URL.

## What happens on build

1. Node stage builds the frontend to `frontend/dist`.
2. Python stage installs dependencies and **trains the model** — about a second,
   which keeps a 5 MB binary out of version control and guarantees the deployed
   model matches the deployed code.
3. Uvicorn serves the API on `/api/*` and the built app on everything else.

## Managing it

```bash
hf spaces logs <user>/cerealia --follow      # live logs
hf spaces restart <user>/cerealia            # restart
hf spaces secrets add <user>/cerealia --secrets GROK_API_KEY=...
hf spaces settings <user>/cerealia --sleep-time 0    # never sleep
```

## Without an API key

The Space still runs. The chatbot falls back to local retrieval over the scheme
database and speech falls back to the browser recogniser — both answer in Hindi.
Everything else is unaffected, since the model, the recommendation engine and
all the data are local to the container.
