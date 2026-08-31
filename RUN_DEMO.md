# Demo runbook

Two terminals. Nothing here needs the internet.

## 1. Backend
```bash
cd ~/minor_proj/backend
./.venv/bin/python -m uvicorn app.main:app --port 8010
```
Check: <http://localhost:8010/api/health> → `"status":"ok"`, `"model_trained":true`

If the model is missing: `./.venv/bin/python train.py` (~1 second).

## 2. Frontend
```bash
cd ~/minor_proj/frontend
npm run dev
```
Opens <http://localhost:5173>

## Demo path (about 3 minutes)

1. **Globe** — spinning, India highlighted. Click **Explore India**.
2. **Map** — 36 states coloured by their top crop type. Hover a few.
3. **Click Maharashtra** — Mango, Papaya, Tur. Point out the MSP badge and
   the "moderate confidence" label.
4. **Expand the top card** — ideal-band comparison, cost breakdown,
   fertiliser plan in bags and rupees.
5. **Drag the farm-size slider** — every rupee figure rescales.
6. **Click Kerala** — completely different answer: banana, rice, coconut,
   coffee. Shows it is regional, not one global ranking.
7. **Click Punjab** — paddy scores 0.39 fitness with a 346 mm irrigation
   gap. This is the water-crisis finding; it is the strongest talking point.
8. **Chat** — "How do I insure my crop against drought?" → retrieves PMFBY.

## If something breaks

- **Panel says "Could not reach the API"** → backend is not running, or
  port 8010 is taken. Check with `lsof -nP -iTCP:8010 -sTCP:LISTEN`.
- **Port 8000 is used by another project on this machine** — that is why
  this one runs on 8010.
- **Chatbot says "Local scheme database"** → expected without a Grok API
  key. It still answers correctly; the fallback is deliberate.
- **Map is blank** → hard-reload the browser; the GeoJSON is bundled
  locally so it cannot be a network problem.

## Numbers worth remembering

| | |
|---|---|
| Dataset | 2,200 rows · 22 crops · exactly 100 each |
| Model | RandomForest, 300 trees |
| Hold-out accuracy | 99.55% |
| 5-fold CV | 99.50% ± 0.27% |
| Errors | 2 of 440 test samples |
| Top features | rainfall 0.22, humidity 0.21, K 0.18 |
| States covered | 36 |
| Schemes indexed | 12 |
| India map | 4 MB → 85 KB, bundled offline |
