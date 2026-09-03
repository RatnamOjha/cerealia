# Demo runbook

## One command

```bash
cd ~/minor_proj && ./dev.sh
```

Starts the API and the web app together and prints a status board:

```
Cerealia
─────────────────────────────────────────
  ✓ API        http://localhost:8010
  ✓ Model      99.5% CV
  ✓ Chatbot    Groq · openai/gpt-oss-120b
  ✓ Speech     Groq · whisper-large-v3-turbo
─────────────────────────────────────────
  Open  http://localhost:5173
```

Ctrl-C stops both. On a fresh clone it also creates the virtual environment,
installs dependencies and trains the model — all skipped when already present.

A `○` instead of `✓` next to Chatbot or Speech means no API key was found; the
app still works, falling back to the local scheme database and the browser
recogniser. Ports are configurable: `API_PORT=8011 WEB_PORT=5174 ./dev.sh`.

## Demo path (about 3 minutes)

1. **Globe** — spinning, India highlighted. Click **Explore India**.
2. **Map** — 36 states coloured by their top crop type. Hover a few.
3. **Click Maharashtra** — Mango, Orange, Banana: Konkan, Nagpur and Jalgaon.
   Point out the MSP badge and the "moderate confidence" label.
4. **Expand the top card** — ideal-band comparison, cost breakdown, fertiliser
   plan in bags and rupees, and the **Data sources** block showing which yield
   figure was used and the crop's share of sown area. This is the slide-worthy
   detail: every number is traceable.
5. **Drag the farm-size slider** — every rupee figure rescales.
6. **Click Kerala** — completely different answer: rice, banana, coconut.
   Coconut is 72% of Kerala's sown area in the official data.
7. **Click Punjab** — paddy scores 0.39 fitness with a 346 mm irrigation
   gap. This is the water-crisis finding; it is the strongest talking point.
8. **Chat in Hindi** — open the advisor (it opens in हिंदी by default) and tap
   the 🎙 mic, then say:
   *"सूखे से मेरी फसल बर्बाद हो गई, बीमा कैसे मिलेगा?"*
   It transcribes, answers in Hindi with PMFBY, and **▶ सुनें** reads the answer
   aloud in Hindi. Toggle to **EN** to show the same question in English.

   If the room's wifi is unreliable, type the Hindi question instead — only
   dictation needs the network. The Hindi answer and the read-aloud are local.

## If something breaks

- **Panel says "Could not reach the API"** → backend is not running, or
  port 8010 is taken. Check with `lsof -nP -iTCP:8010 -sTCP:LISTEN`.
- **Port 8000 is used by another project on this machine** — that is why
  this one runs on 8010.
- **Chatbot says "Local scheme database"** → expected without a Grok API
  key. It still answers correctly, in Hindi or English; the fallback is deliberate.
- **Mic does nothing** → Chrome or Edge only, and the page needs microphone
  permission. Safari will not do `hi-IN` dictation. Voice input also needs a
  network connection (Chrome sends audio to Google); typing the Hindi question
  works offline.
- **Read-aloud sounds wrong** → the device needs a Hindi voice. This Mac has
  **Lekha (hi_IN)** installed and Chrome also exposes **Google हिन्दी**, so it
  is fine. A ⚠ next to सुनें means no Hindi voice was found.
- **Map is blank** → hard-reload the browser; the GeoJSON is bundled
  locally so it cannot be a network problem.

## Numbers worth remembering

| | |
|---|---|
| Official GoI data | 246,091 records · 646 districts · 33 states · 19 years |
| Fitness dataset | 2,200 rows · 22 crops · exactly 100 each |
| Model | RandomForest, 300 trees |
| Hold-out accuracy | 99.55% |
| 5-fold CV | 99.50% ± 0.27% |
| Errors | 2 of 440 test samples |
| Top features | rainfall 0.22, humidity 0.21, K 0.18 |
| States covered | 36 |
| Schemes indexed | 12 |
| India map | 4 MB → 85 KB, bundled offline |
| Punjab paddy yield | 3.73 t/ha (vs Jharkhand 0.97 t/ha) |
| Kerala coconut | 72% of state sown area |
| Languages | Hindi + English, voice in and voice out |
| Hindi coverage | All 12 schemes carry full Hindi text |
