<h1 align="center">Cerealia</h1>

<p align="center">
  <strong>AI crop intelligence for Indian farmers: soil, climate, price and risk in one answer.</strong>
</p>

<p align="center">
  <a href="https://cerealia.onrender.com"><img src="https://img.shields.io/badge/live%20demo-cerealia.onrender.com-4fbf7f?style=for-the-badge" alt="Live demo"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React 18">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

<p align="center">
  <img src="docs/screenshots/10-globe.png" alt="Cerealia landing page: a dark globe with India highlighted" width="820">
</p>

Pick a state on the map and Cerealia ranks the crops worth growing there. It
doesn't just ask whether a crop *can* grow. It weighs soil, climate, water,
market price and yield risk, then explains each recommendation. A Hindi and
English voice assistant answers follow-up questions about the crop and about
the government schemes that apply.

*Named for the Cerealia, the Roman festival of Ceres, goddess of grain and
harvest. It's the same root that gives us the word "cereal".*

> **Try it:** [cerealia.onrender.com](https://cerealia.onrender.com). It runs on
> Render's free tier, so if nobody has visited in the last 15 minutes, the first
> page load takes about a minute while the server wakes up.

---

## Highlights

- **Ranks crops by risk-adjusted return** for all 36 states and union
  territories, instead of predicting one crop name
- **Built on 246,091 official Government of India production records** for
  measured state yields, cultivation patterns and 19 years of yield volatility
- **About 96% accurate under ±20% sensor error.** The model is trained on
  noise-augmented data and was chosen by benchmarking four models. A
  conventionally trained random forest drops to 87% on the same noisy inputs
- **Explains every recommendation**: which soil and climate readings fall
  outside the crop's ideal range, and a fertiliser plan in bags and rupees
- **Hindi voice assistant** with Whisper speech-to-text and answers grounded in
  12 central government schemes
- **Degrades instead of breaking.** With no API key or network, the assistant
  falls back to local retrieval and the browser's speech recogniser

## Tech stack

| Layer | Tools |
|---|---|
| Machine learning | scikit-learn (Extra Trees), pandas, NumPy, joblib |
| Backend | Python 3.12, FastAPI, Pydantic, Uvicorn, httpx |
| Frontend | React 18, Vite, d3-geo, TopoJSON |
| AI services | Groq: `gpt-oss-120b` for chat, Whisper large v3 turbo for speech (xAI also supported) |
| Data | Ministry of Agriculture crop production statistics, CACP cost of cultivation, IMD rainfall normals |
| Deployment | Docker (multi-stage build), Render |

---

## Why this is not just a crop classifier

Most crop-recommendation projects stop at a classifier trained on the Kaggle
`Crop_recommendation.csv`: 7 features in, 1 crop name out, ~99% accuracy, done.
That number is misleading, and a single crop name isn't useful advice.

Cerealia treats the classifier as **one input out of four**:

| Layer | Question it answers | Source |
|---|---|---|
| Agro-climatic fitness | Can this crop survive here? | Extra Trees classifier + per-crop percentile envelopes |
| Regional cultivation prior | Is it actually grown here, and at what scale? | **246,091 official GoI production records** |
| Economics | What does it earn per hectare per year? | **Measured state yields** + CACP MSP and A2+FL costs |
| Water and risk | Can the farmer irrigate it, and how volatile is it? | IMD normals + **19 years of measured yield variance** |

Three of the four come from official Government of India data rather than
estimates. See [Data foundation](#data-foundation).

Crops are ranked by **risk-adjusted expected return**: net profit weighted by
agro-climatic fitness, not raw profit. Ranking on profit alone makes the system
recommend pomegranate for all 36 states, including Punjab, where it scores 0%
suitability. You can reproduce that failure by setting
`DEFAULT_WEIGHTS["profit"] = 0.9` in `recommender.py`.

## What the engine gets right

The rankings match real Indian cropping patterns, and none of them are
hard-coded:

| State | Top 3 recommendations | Reality check |
|---|---|---|
| Assam | Jute, Rice, Coconut | The jute and paddy belt |
| West Bengal | Jute, Papaya, Rice | India's largest jute producer |
| Kerala | Rice, Banana, Coconut | Coconut is 72% of Kerala's sown area |
| Maharashtra | Mango, Pomegranate, Orange | Konkan mango, Solapur pomegranate, Nagpur orange |
| Punjab | Maize, Watermelon, Cotton | Cotton is 15% of Punjab's sown area |
| Rajasthan | Watermelon, Pomegranate, Moth beans | Moth beans are 22% of Rajasthan's sown area |
| Himachal Pradesh | Maize, Orange, Mango | Maize is 76% of Himachal's sown area |

It also surfaces real agronomic tension instead of hiding it. **Punjab paddy
scores only 35% agro-climatic fit**, because 649 mm of annual rainfall can't
support a crop that needs about 1,200 mm. Punjab grows it anyway (79% of its
sown area) by pumping groundwater, and the engine flags a **746 mm irrigation
gap**. That's the state's documented water crisis showing up in the output.

---

## Features

### Recommendation engine

Each state comes with an agro-climatic profile: soil N, P, K and pH,
temperature, humidity and rainfall. The engine scores all 22 crops against it,
combines the four signals above, and returns a ranked list with net profit per
hectare, expected profit after risk, water need and a confidence level. The map
colours every state by the type of crop ranked first there.

### Explainability

Every recommendation compares each reading with the range that crop is
observed to thrive in (its 10th–90th percentile band), mismatches first. For
Kerala's top pick, rice: *phosphorus is 30 kg/ha, below rice's ideal 36–58
kg/ha.* That gap then becomes a fertiliser plan: about 38 kg/ha of DAP, roughly
₹1,030 per hectare at subsidised rates.

### Hindi and voice

Most Indian farmers aren't fluent in English, so an English-only interface
makes everything above unusable. The assistant works in Hindi end to end:

- **Speech input via Whisper.** The browser records audio with `MediaRecorder`,
  which works in every major browser, and the backend transcribes it with
  Whisper large v3 turbo on Groq. The API key never reaches the browser.
- **Hindi answers.** Devanagari in the question routes the reply to Hindi, or
  the farmer can pin the language with the toggle. Each language keeps its own
  conversation, so switching never shows a Hindi answer under English labels.
- **Read aloud** through the device's own speech synthesis.
- **Guarded language.** If the LLM answers a Hindi question in English, that
  reply is discarded and a Hindi template is served instead.

### Scheme and crop assistant

The assistant answers two kinds of question:

- **"What should I grow?"** It answers from the engine's ranking for the
  selected state, so the advice matches the panel on screen.
- **Government schemes.** It does grounded retrieval over 12 curated central
  schemes: PM-KISAN, PMFBY, KCC, Soil Health Card, PMKSY, PM-KUSUM, e-NAM, MSP,
  MIDH, FPO, AIF and Natural Farming. The retrieved scheme text is passed to
  the LLM, which is told to answer only from it, because a made-up subsidy
  percentage could cost a farmer real money.

**Without an API key**, the same retrieval renders a structured answer
directly, so the feature degrades to search instead of going down. Hindi
keywords are matched as substrings, because Devanagari words change their
endings: `बीमा` has to match inside `बीमे का`.

---

## Model

- **`ExtraTreesClassifier`**, 150 trees with at least 8 samples per leaf, on 7
  features and 22 crops
- **Noise-augmented training.** Each training row gets four copies with
  5–30%-scale random error added, so the model learns to handle imperfect
  field readings rather than lab-perfect ones
- **Headline: 95.7% on held-out data with ±20% simulated sensor error.** Clean
  hold-out accuracy is 98.9%, and 5-fold cross-validation gives 98.7% ± 0.4%.
  A pristine reading isn't what a farmer's NPK test strip produces, so the
  noisy figure is the one worth quoting. The model is retrained on every image
  build, so the exact number can move slightly between platforms; the live
  figure is served at `/api/health`

**Chosen by benchmark, not assumption.** Five-fold cross-validated accuracy at
each level of input noise, with augmentation applied inside each fold so no
noisy copy of a test row leaks into training:

| Model | Clean | ±10% | ±20% | ±30% | ±50% |
|---|---|---|---|---|---|
| Random forest, clean training | 99.5% | 95.9% | 86.9% | 75.3% | 56.2% |
| Random forest, noise-augmented | 99.2% | 98.4% | 94.6% | 86.0% | 67.0% |
| **Extra Trees, noise-augmented** | 98.6% | **98.6%** | **95.8%** | **89.2%** | **73.6%** |
| Gradient boosting, noise-augmented | 98.5% | 98.0% | 95.6% | 86.8% | 70.1% |

Extra Trees also keeps `feature_importances_`, which the explainability layer
reads directly; gradient boosting doesn't provide it. Re-run the comparison
with `python backend/train.py --benchmark`.

**Why the accuracy audit matters.** On the clean data, Gaussian Naive Bayes
scores 99.45%, the same as the random forest. When the simplest model
nearly matches the strongest one, the dataset is too easy to tell the models
apart, which is why the clean score says little and the noisy score says more.
`train.py` runs this audit every time.

**Other decisions:**

- **Regularised trees.** Grown to purity, the trees memorised the added noise
  instead of learning from it: a 281 MB model that scored *lower* under noise
  and ran the 512 MB server out of memory on its first request. Stopping at 8
  samples per leaf gives better noisy accuracy at 20 MB, and `train.py` fails
  the build if the model grows past 60 MB.
- **No SMOTE.** The dataset has exactly 100 rows per crop, so oversampling would
  add synthetic rows without fixing any imbalance. `train.py` checks the balance
  instead of assuming it.
- **Top features by importance:** humidity (0.20), rainfall (0.20),
  phosphorus (0.15).

---

## Data foundation

The biggest upgrade to this project was replacing hand-written estimates with
the Ministry of Agriculture's **district-wise, season-wise crop production
statistics**: 246,091 records across 646 districts, 33 states, 124 crops and 19
years (1997–2015), published on data.gov.in.

`backend/build_crop_stats.py` maps the 124 source crop names onto the 22
modelled crops and derives three inputs that used to be guesses:

| Input | Before | Now |
|---|---|---|
| Cultivation prior | Our judgement of what each state grows | Measured sown area: Punjab 79% rice and 15% cotton, Kerala 72% coconut |
| Yield | One national figure per crop | Measured per state: Punjab paddy 3.73 t/ha vs Jharkhand 0.97 t/ha |
| Risk | A hand-assigned 1–5 score | Coefficient of variation of yield over 19 years, blended with market risk |

This corrected our own mistakes too: we had listed banana as Kerala's leading
crop, and the official data says coconut, by a factor of eighteen.

**The official dataset focuses on field crops.** It records no apple at all for
Himachal Pradesh and puts Maharashtra's grapes at 0.03% of sown area. Ranking
on sown area alone would bury high-value crops that use little land, which are
exactly the crops this project should surface. So a curated horticulture list
supplements it, and `regional_prior()` takes the stronger of the two signals for
each crop.

Field crops are costed **per quintal** using CACP A2+FL cost of production,
the same basis the government uses to set MSP at roughly 1.5× cost. A fixed
per-hectare cost tuned against optimistic yields turns paddy into a loss the
moment real yields are used.

---

## Architecture

```
Browser (React + Vite)
  globe  ->  India state map  ->  recommendation panel  ->  voice assistant
                     |                     |                      |
                     +-------------- REST (same origin) ----------+
                                           |
                                FastAPI (backend/app)
            +-----------------+------------+-----------+-------------------+
            |                 |                        |                   |
     recommender.py       chatbot.py               speech.py           train.py
     4-signal ranking     grounded retrieval       Whisper STT         Extra Trees
            |                 |                        |                   |
     state profiles       schemes.json             Groq / xAI      Crop_recommendation.csv
     crop economics       Groq / xAI LLM                            2,200 rows, 22 crops
     cultivation prior
```

In production, one container serves both the API and the built frontend, so
there's no CORS setup and only one service to keep running.

---

## Running locally

```bash
./dev.sh
```

That's all. It creates the virtual environment, installs dependencies, trains
the model, starts the API and the web app, and prints a status board, skipping
any step that's already done. Ctrl-C stops both. Open http://localhost:5173.

**Optional:** enable the LLM assistant and server-side speech recognition.

```bash
cp backend/.env.example .env
# then add your key:  GROK_API_KEY=...
```

The provider is detected from the key's prefix:

| Key prefix | Provider | Chat | Speech-to-text |
|---|---|---|---|
| `gsk_…` | [Groq](https://console.groq.com) | gpt-oss-120b | Whisper large v3 turbo, **free** up to 2,000 a day |
| `xai-…` | [xAI](https://console.x.ai) | grok-3 | xAI STT, billed per hour |

Without a key the app still runs: the assistant uses local retrieval over the
scheme database, and speech uses the browser's recogniser.

## Deployment

Live at **[cerealia.onrender.com](https://cerealia.onrender.com)** on Render's
free plan. It's configured by [`render.yaml`](render.yaml) and redeploys
automatically on every push to `main`.

The multi-stage [`Dockerfile`](Dockerfile) builds the frontend with Node, then
installs the Python dependencies and trains the model during the image build,
so the deployed model always matches the deployed code. The free instance has
512 MB of memory and sleeps after 15 idle minutes. [DEPLOY.md](DEPLOY.md) covers
setup, custom domains, and why Hugging Face Spaces, Netlify and Cloud Run were
ruled out.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status, AI provider and live model metrics |
| `GET /api/states` | All 36 states and UTs with agro-climatic profiles |
| `GET /api/states/{id}` | One state's profile |
| `GET /api/overview` | Top crop for every state, used to colour the map |
| `POST /api/recommend/state` | Ranked crops for a state, with optional soil overrides |
| `POST /api/recommend/custom` | Ranked crops from raw soil and climate readings |
| `GET /api/schemes` | All government schemes |
| `POST /api/chat` | Crop and scheme assistant (`lang`: `auto`, `en` or `hi`) |
| `POST /api/stt` | Speech-to-text for a recorded question |

---

## Known limitations

Stated plainly, because they shape the roadmap:

1. **The fitness dataset is synthetic.** `Crop_recommendation.csv` has clean,
   well-separated ranges for each crop, which is why 99.5% accuracy is easy and
   why that number shouldn't be read as real-world performance. It has no
   geography at all, which is exactly why the official cultivation prior exists.
2. **Apple is under-ranked for Himachal.** The dataset puts apple's temperature
   range at 21–24 °C, which doesn't match real apple farming (apple needs about
   1,000 winter chill hours). A chill-hour feature is needed.
3. **State-level granularity.** Nashik and Vidarbha are both just
   "Maharashtra". District-level resolution is the next step.
4. **Annual averages, not seasons.** Crops grow in specific seasons, and annual
   averages blur kharif and rabi together.
5. **Prices are national indicative figures**, not live district mandi rates.
6. **Wheat and sugarcane are missing.** They aren't among the 22 crops in the
   fitness dataset, even though they appear in the production statistics.
7. **The production data ends in 2015.** Yields and cropping patterns have
   changed since, and the newer release still needs to be added.
8. **Cotton's measured volatility is inflated**, because states report lint and
   kapas in different units. The risk value is capped rather than trusted as is.
9. **Assistant answers are phrased by the LLM.** They're grounded in the
   engine's ranking, but the model writes the wording, so the emphasis can
   differ between an English and a Hindi answer.
10. **Browser speech fallback needs a network.** Without an API key, dictation
    uses the browser's recogniser, which only works in Chrome and Edge and sends
    audio to Google. [Bhashini](https://bhashini.gov.in) would remove that
    dependency and add 21 more Indian languages.

### A unit bug worth recording

The dataset's `rainfall` column is per growing cycle (20–299 mm), while IMD
state rainfall normals are annual totals (108–3,062 mm). Feeding annual figures
straight into the model pushed every state far outside the training data, the
model returned near-zero probability for everything, and profit won by default,
producing "pomegranate for all 36 states". The model input is now
`rainfall_annual_mm / 12`, and the annual figure is kept separately for the
irrigation-gap calculation.

---

## Roadmap

**Near term:** district-level resolution; live weather from Open-Meteo; live
mandi prices from Agmarknet; SHAP explanations; Bhashini for 21 more languages
and offline dictation.

**Medium term:** crop rotation planning across seasons; groundwater
sustainability scoring from CGWB block data; mandi price forecasting;
leaf-disease detection from photos; an ESP32 and NPK sensor for live soil
readings; a WhatsApp bot for farmers without smartphones.

**Longer term:** Sentinel-2 NDVI crop health monitoring; land allocation across
multiple crops under a budget; dashboards for FPOs and district officers to see
aggregate planting intentions and warn of gluts early.

---

## Project layout

```
backend/
  app/
    main.py              FastAPI routes; serves the built frontend in production
    recommender.py       4-signal ranking engine
    chatbot.py           grounded crop and scheme assistant
    speech.py            server-side speech-to-text
    config.py            .env loading and AI provider detection
    data/                datasets, state profiles, economics, priors, schemes
  train.py               model training, accuracy audit and benchmark
  build_crop_stats.py    derives state yields, risk and cultivation prior from GoI data
  models/                trained model and metrics (model is regenerated by train.py)
frontend/                React + Vite client
docs/                    project review deck and screenshots
Dockerfile               single-container build: frontend + API + model
render.yaml              Render deployment config
dev.sh                   one-command local setup and run
DEPLOY.md                deployment guide
```
