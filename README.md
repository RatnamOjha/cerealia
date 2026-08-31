# KrishiMitra — AI Crop Recommendation & Advisory for Indian Farmers

An interactive platform where a farmer picks their region on a 3D globe / map of
India and gets a ranked set of crops to grow — scored not just on whether the
crop *can* grow there, but on what it is actually worth growing, given soil,
climate, water availability, market price and risk. A grounded AI assistant then
answers questions about the government schemes that apply.

> Minor project. Status: backend and recommendation engine working end to end;
> frontend in progress.

---

## Why this is not just a crop classifier

Most crop-recommendation projects stop at a classifier over the Kaggle
`Crop_recommendation.csv` — 7 features in, 1 crop name out, ~99% accuracy,
done. That number is misleading and the output is not useful advice.

This project treats the classifier as **one input out of four**:

| Layer | Question it answers | Source |
|---|---|---|
| Agro-climatic fitness | Can this crop survive here? | RandomForest + per-crop percentile envelopes |
| Regional cultivation prior | Is this crop actually grown in this region? | Agricultural Statistics at a Glance, NHB/APEDA |
| Economics | What does it earn per hectare per year? | CACP MSP 2024-25, Agmarknet modal prices, cost of cultivation |
| Water & risk | Can the farmer irrigate it, and how volatile is it? | Crop water requirement vs IMD rainfall normals |

Crops are ranked by **risk-adjusted expected return** — net profit weighted by
agro-climatic fitness — not by headline profit. Ranking on raw profit alone
makes the system recommend pomegranate for all 36 states, including Punjab where
it scores 0% suitability. That failure is reproducible: set
`DEFAULT_WEIGHTS["profit"] = 0.9` in `recommender.py` and re-run.

---

## What the engine gets right

Recommendations reproduce real Indian cropping patterns without any of them
being hard-coded as answers:

| State | Top recommendations | Reality check |
|---|---|---|
| Assam | Jute, Rice | India's jute and paddy belt |
| West Bengal | Jute, Rice, Banana | Largest jute producer |
| Kerala | Banana, Rice, Coconut, Coffee | Matches actual cropping pattern |
| Maharashtra | Mango, Tur, Orange | Konkan mango, Vidarbha tur, Nagpur orange |
| Rajasthan | Moth beans, Watermelon, Chickpea | Arid-zone crops |
| Himachal Pradesh | Maize, Orange, Rajma | Maize is HP's largest crop by area |

It also surfaces genuine agronomic tension rather than hiding it. **Punjab paddy
scores only 0.39 agro-climatic fitness** — because 649 mm annual rainfall cannot
support rice. Punjab grows it anyway via groundwater extraction, and the engine
flags a 346 mm irrigation gap. That is the state's documented water crisis
appearing directly in the model output.

---

## Architecture

```
Browser (React + Vite)
  3D globe  ->  India state map  ->  recommendation panel  ->  scheme chatbot
        |                                    |                       |
        +------------------ REST ------------+-----------------------+
                                   |
                          FastAPI (backend/app)
                    |              |                 |
            recommender.py    chatbot.py        train.py
            4-signal rank    grounded RAG     RandomForest
                    |              |                 |
              state profiles   schemes.json    Crop_recommendation.csv
              crop economics   Grok (x.ai)     2200 rows / 22 crops
              cultivation prior
```

### Model

- `RandomForestClassifier(n_estimators=300)` on 7 features, 22 crops
- Hold-out accuracy **0.9955**, 5-fold CV **0.9950 ± 0.0027**, 2 errors in 440
- **No SMOTE.** The dataset is exactly balanced at 100 rows per class, so
  resampling would inject synthetic noise without correcting any imbalance.
  `train.py` asserts the balance rather than assuming it.
- Top features by importance: rainfall (0.22), humidity (0.21), K (0.18)

### Explainability

Every recommendation returns a per-feature comparison against that crop's
observed 10th–90th percentile band — *"your rainfall of 244 mm sits inside
rice's ideal 182–298 mm band"* — sorted so mismatches surface first. Plus a
fertiliser plan converting the NPK gap into bags of urea/DAP/MOP and a rupee
cost.

### Chatbot

Grounded retrieval over 12 curated central schemes (PM-KISAN, PMFBY, KCC,
Soil Health Card, PMKSY, PM-KUSUM, e-NAM, MSP, MIDH, FPO, AIF, Natural Farming).
Retrieved scheme text is passed to Grok as context and the model is instructed to
answer only from it — a hallucinated subsidy percentage could cost a farmer real
money. **Without an API key the same retrieval renders a structured answer
directly**, so the feature degrades to a search system rather than going down,
and the demo runs with no network.

---

## Running it

```bash
# Backend
cd backend
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python train.py                 # ~1s, writes models/
./.venv/bin/python -m uvicorn app.main:app --reload --port 8010

# Frontend
cd frontend && npm install && npm run dev    # http://localhost:5173
```

Optional — enable the Grok-backed chatbot:

```bash
cp backend/.env.example backend/.env
# add GROK_API_KEY from https://console.x.ai
```

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status + live model metrics |
| `GET /api/states` | All 36 states/UTs with agro-climatic profiles |
| `POST /api/recommend/state` | Ranked crops for a state (+ optional soil overrides) |
| `POST /api/recommend/custom` | Ranked crops from raw soil/climate readings |
| `GET /api/schemes` | All government schemes |
| `POST /api/chat` | Scheme advisory chatbot |

---

## Known limitations

Stated plainly, because they shape the roadmap:

1. **The training dataset is synthetic.** `Crop_recommendation.csv` has clean,
   well-separated per-crop feature envelopes — which is why 99.5% accuracy is
   easy and why that number should not be read as real-world performance. It
   carries no geography at all, which is precisely why the cultivation prior
   layer exists.
2. **Apple is under-ranked for Himachal.** The dataset places apple's temperature
   band at 21–24 °C, which does not reflect real apple agronomy (apple needs
   ~1000 winter chill hours). A vernalisation/chill-hour feature is needed.
3. **State-level granularity.** Nashik and Vidarbha are both "Maharashtra" here.
   District-level resolution is the next iteration.
4. **Annual averages, not seasonal.** Crops grow in specific seasons; annual
   means blur kharif and rabi together.
5. **Prices are national indicative figures**, not live district mandi rates.
6. **Wheat and sugarcane are absent** — they are simply not among the 22 crops in
   the dataset, despite being two of India's largest crops.

### One unit bug worth recording

The dataset's `rainfall` column ranges 20–299 mm (per growing cycle), while IMD
state rainfall normals are annual totals (108–3062 mm). Feeding annual figures
straight into the model pushed every state far outside the training distribution,
the forest returned ≈0 probability for everything, and the profit term won by
default — producing "pomegranate for all 36 states". The model input is now
derived as `rainfall_annual_mm / 12`, with the annual figure retained separately
for the irrigation-gap calculation.

---

## Roadmap

**Near term** — district-level resolution; live weather via Open-Meteo; live mandi
prices via Agmarknet/data.gov.in; SHAP explanations; Hindi + regional languages
via Bhashini.

**Medium term** — crop rotation planning across seasons; groundwater
sustainability scoring from CGWB block data; mandi price forecasting;
leaf-disease detection from photos; ESP32 + NPK sensor for live soil readings;
WhatsApp bot for farmers without smartphones.

**Longer term** — Sentinel-2 NDVI crop health monitoring; land-allocation
optimisation across multiple crops under budget constraints; FPO/district
officer dashboards for aggregate cropping intent and glut early-warning.

---

## Layout

```
backend/
  app/
    main.py           FastAPI routes
    recommender.py    4-signal ranking engine
    chatbot.py        grounded scheme advisory
    data/             dataset, state profiles, economics, cultivation prior, schemes
  train.py            model training + evaluation
  models/             trained artefacts (gitignored, regenerate with train.py)
frontend/             React + Vite client
hardware/             ESP32 soil sensor firmware (planned)
docs/                 presentation and design notes
```
