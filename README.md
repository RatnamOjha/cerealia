# Cerealia — AI Crop Intelligence for Farmers

*Named for the Roman festival of Ceres, goddess of grain and harvest — the same root that gives us the word “cereal”.*

An interactive platform where a farmer picks their region on a 3D globe and gets
a ranked set of crops to grow — scored not just on whether the
crop *can* grow there, but on what it is actually worth growing, given soil,
climate, water availability, market price and risk. A grounded AI assistant then
answers questions about the government schemes that apply.

> Minor project. Status: working end to end — model, ranking engine, REST API,
> React interface and scheme chatbot. Runs fully offline. See [RUN_DEMO.md](RUN_DEMO.md).

---

## Why this is not just a crop classifier

Most crop-recommendation projects stop at a classifier over the Kaggle
`Crop_recommendation.csv` — 7 features in, 1 crop name out, ~99% accuracy,
done. That number is misleading and the output is not useful advice.

This project treats the classifier as **one input out of four**:

| Layer | Question it answers | Source |
|---|---|---|
| Agro-climatic fitness | Can this crop survive here? | RandomForest + per-crop percentile envelopes |
| Regional cultivation prior | Is it actually grown here, and at what scale? | **246,091 official GoI production records** |
| Economics | What does it earn per hectare per year? | **Measured state yields** + CACP MSP and A2+FL costs |
| Water & risk | Can they irrigate it, and how volatile is it? | IMD normals + **19 years of measured yield variance** |

Three of those four are derived from official Government of India data rather
than estimated. See [Data foundation](#data-foundation).

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
| Assam | Jute, Rice | The jute and paddy belt |
| West Bengal | Jute, Rice, Banana | India's largest jute producer |
| Kerala | Rice, Banana, Coconut | Coconut is 72% of Kerala's sown area |
| Maharashtra | Mango, Orange, Banana | Konkan mango, Nagpur orange, Jalgaon banana |
| Punjab | Maize, Cotton, Lentil | Cotton is 15% of Punjab's sown area |
| Rajasthan | Watermelon, Moth beans | Moth beans are 22% of Rajasthan's area |
| Himachal Pradesh | Maize, Orange, Rajma | Maize is 76% of HP's sown area |

It also surfaces genuine agronomic tension rather than hiding it. **Punjab paddy
scores only 0.39 agro-climatic fitness** — because 649 mm annual rainfall cannot
support rice. Punjab grows it anyway via groundwater extraction, and the engine
flags a 346 mm irrigation gap. That is the state's documented water crisis
appearing directly in the model output.

---

## Data foundation

The single largest upgrade to this project was replacing hand-authored estimates
with the Ministry of Agriculture's **district-wise, season-wise crop production
statistics** — 246,091 records across 646 districts, 33 states, 124 crops and 19
years (1997–2015), published on data.gov.in.

`backend/build_crop_stats.py` maps 124 source crop names onto the 22 modelled
crops and derives three inputs that were previously guesses:

| Input | Was | Now |
|---|---|---|
| Cultivation prior | Our judgement of what each state grows | Measured sown area — Punjab 79% rice / 15% cotton, Kerala 72% coconut |
| Yield | One national figure per crop | State-specific and measured — Punjab paddy 3.73 t/ha vs Jharkhand 0.97 t/ha |
| Risk | A hand-assigned 1–5 score | Coefficient of variation of yield across 19 years, blended with market risk |

This corrected our own errors, not just the model's: we had listed banana as
Kerala's leading crop; the official data says coconut, by a factor of eighteen.

**The official dataset is field-crop focused.** It records no apple for Himachal
Pradesh at all, and puts Maharashtra's grapes at 0.03% of sown area. Ranking on
sown area alone structurally buries high-value crops that occupy little land —
exactly the class of crop this project exists to surface. So a curated
horticulture list is retained as a supplement, and `regional_prior()` takes the
stronger of the two signals per crop.

Field crops are costed **per quintal** using CACP A2+FL cost of production, the
same basis the government uses to set MSP at roughly 1.5× cost. A fixed
per-hectare cost calibrated against optimistic yields turns paddy into a phantom
loss the moment real yields are substituted in.

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

- `ExtraTreesClassifier(n_estimators=300)` on 7 features, 22 crops, trained on
  noise-augmented data so it holds up on real instrument readings
- **~95% accuracy under ±20% sensor error** — the figure worth quoting, measured
  on held-out data the model never saw. Clean hold-out is 0.9932 and 5-fold CV
  0.9932 ± 0.0020, but a pristine reading is not what a farmer's NPK strip
  produces. The model is retrained on every image build, so the exact figure
  moves a little with the platform's BLAS; the live one is served at
  `/api/health`
- Selected by benchmark, not assumption. At ±20% noise: extra trees 96.1%,
  gradient boosting 95.6%, noise-augmented forest 95.4%, clean-trained forest
  88.1%. Boosting also drops `feature_importances_`, which the explainability
  layer reads directly
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

### Hindi and voice

Most Indian farmers are not fluent in English, which makes the language barrier
the real adoption barrier — every feature above is unusable if the interface
only speaks English.

- **Speech input** in Hindi via the Web Speech API (`hi-IN`). The transcript
  submits itself, so voice is a single action rather than dictate-then-press-send.
- **Hindi answers.** Devanagari in the question routes the reply to Hindi, or the
  farmer can pin the language with the toggle. All 12 schemes carry Hindi text
  (`name_hi`, `benefit_hi`, `eligibility_hi`, `how_to_apply_hi`) plus Hindi
  retrieval keywords, matched as substrings because Devanagari inflects with
  suffixes — `बीमा` has to match inside `बीमे का`.
- **Read aloud** via speech synthesis using the device's own Hindi voice.
- If Grok answers a Hindi question in English, the response is discarded and the
  Hindi template served instead. A farmer cannot work around a wrong-language
  reply.

**Honest limitation:** Chrome's recogniser streams audio to Google's servers, so
*dictation* needs a network connection even though everything else here runs
offline. Speech output is fully local. Moving dictation to
[Bhashini](https://bhashini.gov.in) removes that dependency and adds 21 more
scheduled languages.

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
./dev.sh
```

That is the whole thing. It creates the virtual environment, installs
dependencies, trains the model, starts the API and the web app, and prints a
status board — skipping any step already done. Ctrl-C stops both.

Optional — enable the AI chatbot and server-side speech recognition:

```bash
cp backend/.env.example .env
# paste your key:  GROK_API_KEY=...
```

The provider is detected from the key prefix, so either works:

| Key prefix | Provider | Chat | Speech-to-text |
|---|---|---|---|
| `gsk_…` | [Groq](https://console.groq.com) | gpt-oss-120b | Whisper large v3 turbo — **free**, 2,000/day |
| `xai-…` | [xAI](https://console.x.ai) | grok-3 | xAI STT — billed per hour |

Without a key the app still runs: the chatbot falls back to local retrieval over
the scheme database and speech falls back to the browser recogniser.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status + live model metrics |
| `GET /api/states` | All 36 states/UTs with agro-climatic profiles |
| `POST /api/recommend/state` | Ranked crops for a state (+ optional soil overrides) |
| `POST /api/recommend/custom` | Ranked crops from raw soil/climate readings |
| `GET /api/schemes` | All government schemes |
| `POST /api/chat` | Scheme advisory chatbot (`lang`: `auto` / `en` / `hi`) |

---

## Known limitations

Stated plainly, because they shape the roadmap:

1. **The fitness dataset is synthetic.** `Crop_recommendation.csv` has clean,
   well-separated per-crop feature envelopes — which is why 99.5% accuracy is
   easy and why that number should not be read as real-world performance. It
   carries no geography at all, which is precisely why the official cultivation
   prior exists.
2. **Apple is under-ranked for Himachal.** The dataset places apple's temperature
   band at 21–24 °C, which does not reflect real apple agronomy (apple needs
   ~1000 winter chill hours). A vernalisation/chill-hour feature is needed.
3. **State-level granularity.** Nashik and Vidarbha are both "Maharashtra" here.
   District-level resolution is the next iteration.
4. **Annual averages, not seasonal.** Crops grow in specific seasons; annual
   means blur kharif and rabi together.
5. **Prices are national indicative figures**, not live district mandi rates.
   MSP is national by definition; mandi prices vary by district and by week.
6. **Wheat and sugarcane are absent** — they are simply not among the 22 crops in
   the fitness dataset, despite appearing in the production statistics.
7. **The production series ends in 2015.** Yields and cropping patterns have moved
   since; the newer release needs ingesting.
8. **Cotton's measured volatility is inflated** because states report lint and
   kapas in different units. The risk conversion is capped rather than trusted
   unbounded.

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
prices via Agmarknet/data.gov.in; SHAP explanations; Bhashini for the remaining
21 scheduled languages and on-device (offline) dictation.

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
    data/             datasets, state profiles, economics, priors, schemes
  train.py            model training + evaluation
  build_crop_stats.py derives state yields, risk and cultivation prior from GoI data
  models/             trained artefacts (gitignored, regenerate with train.py)
frontend/             React + Vite client
hardware/             ESP32 soil sensor firmware (planned)
docs/                 presentation and design notes
```
