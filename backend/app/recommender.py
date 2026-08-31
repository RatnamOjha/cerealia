"""
Crop recommendation engine.

The ML classifier only tells us what *can* grow. A farmer needs to know what is
worth growing. This module composes four signals into one ranked answer:

  1. Agro-climatic fitness -- how well soil + climate match the crop's envelope
  2. Expected net profit   -- yield x price - cost, annualised for perennials
  3. Water feasibility     -- crop water demand vs local rainfall
  4. Risk                  -- price volatility, perishability, capital lock-in

Fitness is a *gate*, not just another weighted term. A crop that cannot survive
the local climate is removed outright, however profitable it looks on paper --
otherwise the ranking degenerates into "plant pomegranate everywhere", which is
exactly what a naive profit-weighted score does.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crop_suitability.joblib"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# The classifier was trained on per-growing-cycle rainfall (20-299 mm), while
# state profiles carry annual totals (108-3062 mm). Feeding an annual figure
# straight in pushes every state far outside the training distribution and the
# forest returns ~0 probability for everything.
MONTHS_PER_YEAR = 12.0

PRODUCTIVE_YEARS = {
    "mango": 30, "coconut": 40, "coffee": 25, "orange": 15, "apple": 25,
    "grapes": 15, "pomegranate": 12, "banana": 3, "papaya": 3,
}

FERTILISERS = {
    "Urea":  {"nutrient": "N", "fraction": 0.46, "price_per_kg": 5.6},
    "DAP":   {"nutrient": "P", "fraction": 0.46, "price_per_kg": 27.0},
    "MOP":   {"nutrient": "K", "fraction": 0.60, "price_per_kg": 17.5},
}

DEFAULT_WEIGHTS = {
    "fitness": 0.45,
    "profit": 0.30,
    "water": 0.15,
    "risk": 0.10,
}

# Prior weights for crops the region demonstrably supports.
PRIOR_MAJOR = 1.0          # >= 5% of the state's sown area, or a listed major crop
PRIOR_MINOR_STATS = 0.7    # 0.1-5% of sown area in official statistics
PRIOR_MINOR = 0.6          # listed as a minor crop of the state
PRIOR_NEGLIGIBLE = 0.4     # present in the record but under 0.1% of area

# Empirical yield variability above which a crop counts as maximally risky.
# Cotton's coefficient of variation is inflated by states reporting lint in
# different units, so the conversion is capped rather than trusted unbounded.
RISK_CV_CAP = 0.8

# Agro-climatic floor, applied to fitness *before* the regional prior. When a
# prior is available it does the hard filtering, so this only needs to catch
# crops grown in the state that are a poor match for these specific readings.
# Deliberately permissive: a crop can be viable under irrigation even when
# rainfall alone would not support it, and the water assessment flags that
# separately rather than hiding the option.
FITNESS_GATE = 0.25
MIN_RESULTS = 3


@dataclass(frozen=True)
class SiteConditions:
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float          # per-growing-cycle mm, the model's native unit
    rainfall_annual_mm: float = 0.0

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([[getattr(self, f) for f in FEATURES]], columns=FEATURES)

    @property
    def annual_rain(self) -> float:
        return self.rainfall_annual_mm or self.rainfall * MONTHS_PER_YEAR


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python backend/train.py` first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _economics() -> dict[str, Any]:
    return json.loads((DATA_DIR / "crop_economics.json").read_text())["crops"]


@lru_cache(maxsize=1)
def _states() -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / "state_profiles.json").read_text())["states"]


@lru_cache(maxsize=1)
def _calendar() -> dict[str, Any]:
    return json.loads((DATA_DIR / "state_crop_calendar.json").read_text())["states"]


@lru_cache(maxsize=1)
def _crop_stats() -> dict[str, Any]:
    """Per-state crop statistics derived from official GoI area/production data."""
    return json.loads((DATA_DIR / "state_crop_stats.json").read_text())


def state_stats(state_id: str | None) -> dict[str, Any]:
    if not state_id:
        return {}
    return _crop_stats()["states"].get(state_id.upper(), {})


def regional_prior(state_id: str | None) -> dict[str, float] | None:
    """Prior weight per crop for a state; None means "no regional constraint".

    Two sources are combined, and the stronger signal wins:

      1. Official area statistics (246k records, 1997-2015). Authoritative for
         field crops -- if a state sows 79% of its area to rice, that is a fact,
         not an opinion.
      2. A curated horticulture list. The official dataset is field-crop
         focused: it records no apple for Himachal at all, and puts Maharashtra
         grapes at 0.03% of area. Ranking on sown area alone structurally buries
         high-value crops that occupy little land, which is precisely the class
         of crop this project exists to surface.
    """
    if not state_id:
        return None

    prior: dict[str, float] = {}

    for crop, rec in state_stats(state_id).items():
        tier = rec.get("tier")
        if tier == "major":
            prior[crop] = PRIOR_MAJOR
        elif tier == "minor":
            prior[crop] = PRIOR_MINOR_STATS
        elif tier == "negligible":
            prior[crop] = PRIOR_NEGLIGIBLE

    entry = _calendar().get(state_id.upper())
    if entry:
        for crop in entry.get("major", []):
            prior[crop] = max(prior.get(crop, 0.0), PRIOR_MAJOR)
        for crop in entry.get("minor", []):
            prior[crop] = max(prior.get(crop, 0.0), PRIOR_MINOR)

    return prior or None


def prior_source(state_id: str | None, crop: str) -> str:
    """Where this crop's regional evidence came from, for display."""
    rec = state_stats(state_id).get(crop)
    in_cal = False
    entry = _calendar().get((state_id or "").upper())
    if entry:
        in_cal = crop in entry.get("major", []) or crop in entry.get("minor", [])
    if rec and in_cal:
        return "official statistics + horticulture listing"
    if rec:
        return "official area statistics"
    return "horticulture listing"


@lru_cache(maxsize=1)
def _feature_importance() -> dict[str, float]:
    clf = _bundle()["pipeline"].named_steps["clf"]
    return dict(zip(FEATURES, (float(v) for v in clf.feature_importances_)))


@lru_cache(maxsize=1)
def _crop_envelopes() -> dict[str, dict[str, dict[str, float]]]:
    """Per-crop 10th-90th percentile band for every feature.

    This is our explainability layer and our feasibility gate. Rather than a
    black-box score we can say "your rainfall of 244 mm sits inside rice's ideal
    182-298 mm band" -- something an extension officer can verify.
    """
    df = pd.read_csv(DATA_DIR / "Crop_recommendation.csv")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for crop, group in df.groupby("label"):
        out[crop] = {
            f: {
                "low": float(group[f].quantile(0.10)),
                "high": float(group[f].quantile(0.90)),
                "mean": float(group[f].mean()),
                "std": float(group[f].std()) or 1.0,
            }
            for f in FEATURES
        }
    return out


def list_states() -> list[dict[str, Any]]:
    return _states()


def get_state(state_id: str) -> dict[str, Any] | None:
    return next((s for s in _states() if s["id"].upper() == state_id.upper()), None)


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _envelope_fit(crop: str, site: SiteConditions) -> float:
    """Importance-weighted agreement between the site and the crop's ideal bands.

    Inside the 10-90 band scores 1.0. Outside, the score decays smoothly with
    how many standard deviations away the reading is, so "slightly too dry"
    is penalised far less than "arctic".
    """
    envelope = _crop_envelopes().get(crop)
    if not envelope:
        return 0.0
    importance = _feature_importance()
    total_w = sum(importance.values()) or 1.0

    score = 0.0
    for f in FEATURES:
        band = envelope[f]
        value = getattr(site, f)
        if band["low"] <= value <= band["high"]:
            fit = 1.0
        else:
            edge = band["low"] if value < band["low"] else band["high"]
            deviations = abs(value - edge) / band["std"]
            fit = math.exp(-0.5 * deviations)
        score += importance[f] * fit
    return score / total_w


def resolve_yield(crop: str, econ: dict[str, Any], state_id: str | None) -> dict[str, Any]:
    """Best available yield for this crop in this state, and where it came from.

    Preference order: the state's own measured yield, then the national measured
    yield, then the curated agronomic figure. Measured yields carry real
    regional signal -- Punjab paddy at 3.73 t/ha against Jharkhand at 0.97 t/ha
    is a productivity gap a single national average erases entirely.
    """
    rec = state_stats(state_id).get(crop)
    if rec and rec.get("yield_units_ok") and rec.get("yield_t_ha"):
        return {
            "yield_t_ha": float(rec["yield_t_ha"]),
            "source": "state statistics",
            "years": rec.get("years"),
        }

    national = _crop_stats()["national"].get(crop)
    if national:
        return {
            "yield_t_ha": float(national["yield_t_ha"]),
            "source": "national statistics",
            "years": None,
        }

    return {"yield_t_ha": float(econ["yield_t_ha"]), "source": "curated estimate", "years": None}


def resolve_risk(crop: str, econ: dict[str, Any], state_id: str | None) -> dict[str, Any]:
    """Risk as measured yield volatility, falling back to the curated score.

    The coefficient of variation of annual yield is what "risky crop" actually
    means to a farmer: moth beans at CV 0.72 really do fail some years, and
    grapes at CV 0.06 really are dependable under irrigation.
    """
    rec = state_stats(state_id).get(crop)
    cv = None
    if rec and rec.get("years", 0) and rec["years"] >= 5:
        cv = float(rec["yield_cv"])
    elif (nat := _crop_stats()["national"].get(crop)):
        cv = float(nat["yield_cv"])

    # Production risk and market risk are different things. Measured yield
    # volatility says nothing about whether a perishable crop can be sold:
    # papaya yields reliably (CV 0.19) but rots without a cold chain and
    # crashes in price at a local glut. Blending both stops the model treating
    # a dependable yield as a dependable income.
    market_risk = econ["risk_score"] / 5.0
    if cv is None:
        return {"risk": market_risk, "yield_cv": None, "production_risk": None,
                "market_risk": market_risk, "source": "curated estimate"}

    production_risk = min(1.0, cv / RISK_CV_CAP)
    return {
        "risk": 0.5 * production_risk + 0.5 * market_risk,
        "yield_cv": round(cv, 3),
        "production_risk": round(production_risk, 3),
        "market_risk": round(market_risk, 3),
        "source": "measured volatility + market risk",
    }


def _annual_economics(crop: str, econ: dict[str, Any], state_id: str | None = None) -> dict[str, float]:
    """Revenue and cost put on a common per-hectare-per-year footing.

    Comparing a 70-day mung bean crop against a 30-year mango orchard on raw
    per-cycle profit would be meaningless, so short crops are scaled by how many
    cycles fit in a year and perennials carry an amortised share of their
    establishment cost.
    """
    y = resolve_yield(crop, econ, state_id)
    revenue_per_cycle = y["yield_t_ha"] * 10.0 * econ["price_per_quintal"]
    opex = econ["opex_per_ha"]
    duration = max(econ["duration_days"], 1)

    # Only genuinely short-duration catch crops fit two cycles in a year.
    # Deriving this from duration alone gave jute (120 days) two harvests,
    # which double-counted its revenue.
    cycles_per_year = 2.0 if duration <= 75 and econ["establishment_years"] == 0 else 1.0

    gross = revenue_per_cycle * cycles_per_year

    # Field crops cost roughly in proportion to what they produce, so cost is
    # taken per quintal where CACP publishes it. Horticulture spend is driven by
    # labour and canopy management, so it stays per hectare.
    cost_per_qtl = econ.get("cost_per_quintal")
    if cost_per_qtl:
        running = y["yield_t_ha"] * 10.0 * cost_per_qtl * cycles_per_year
    else:
        running = opex * cycles_per_year

    capex = econ.get("capex_per_ha", 0)
    amortised_capex = capex / PRODUCTIVE_YEARS.get(crop, 15) if capex else 0.0

    net = gross - running - amortised_capex
    return {
        "yield_t_ha_used": round(y["yield_t_ha"], 3),
        "yield_source": y["source"],
        "gross_revenue_per_ha_year": round(gross, 0),
        "operating_cost_per_ha_year": round(running, 0),
        "amortised_capex_per_ha_year": round(amortised_capex, 0),
        "net_profit_per_ha_year": round(net, 0),
        "cycles_per_year": round(cycles_per_year, 2),
    }


def _water_assessment(econ: dict[str, Any], annual_rain_mm: float) -> dict[str, Any]:
    """How much irrigation the crop needs beyond what the sky provides."""
    demand = econ["water_mm"]
    # Roughly 70% of rainfall is effective for crops; the rest runs off or percolates.
    effective_rain = annual_rain_mm * 0.70
    gap = max(0.0, demand - effective_rain)
    ratio = gap / demand if demand else 0.0
    if ratio == 0:
        verdict, label = "rainfed", "Rainfed — no irrigation needed"
    elif ratio < 0.25:
        verdict, label = "light", "Light supplemental irrigation"
    elif ratio < 0.55:
        verdict, label = "moderate", "Moderate irrigation required"
    else:
        verdict, label = "heavy", "Heavy irrigation — check groundwater status"
    return {
        "demand_mm": demand,
        "effective_rainfall_mm": round(effective_rain, 0),
        "irrigation_gap_mm": round(gap, 0),
        "dependence_ratio": round(ratio, 3),
        "verdict": verdict,
        "label": label,
    }


def _explain(crop: str, site: SiteConditions) -> list[dict[str, Any]]:
    """Compare each site reading against the crop's observed ideal band."""
    envelope = _crop_envelopes().get(crop, {})
    pretty = {
        "N": ("Nitrogen", "kg/ha"), "P": ("Phosphorus", "kg/ha"),
        "K": ("Potassium", "kg/ha"), "temperature": ("Temperature", "°C"),
        "humidity": ("Humidity", "%"), "ph": ("Soil pH", ""),
        "rainfall": ("Rainfall (per cycle)", "mm"),
    }
    importance = _feature_importance()
    reasons = []
    for f in FEATURES:
        band = envelope.get(f)
        if not band:
            continue
        value = getattr(site, f)
        name, unit = pretty[f]
        if band["low"] <= value <= band["high"]:
            status, note = "match", "within ideal range"
        elif value < band["low"]:
            status, note = "low", "below ideal range"
        else:
            status, note = "high", "above ideal range"
        reasons.append({
            "feature": f,
            "label": name,
            "unit": unit,
            "value": round(float(value), 2),
            "ideal_low": round(band["low"], 2),
            "ideal_high": round(band["high"], 2),
            "status": status,
            "note": note,
            "importance": round(importance[f], 4),
        })
    # Surface mismatches first -- the farmer needs to see problems, not confirmations.
    reasons.sort(key=lambda r: (r["status"] == "match", -r["importance"]))
    return reasons


def _fertiliser_plan(crop: str, site: SiteConditions) -> dict[str, Any]:
    """Convert the gap between current and ideal soil NPK into bags and rupees."""
    envelope = _crop_envelopes().get(crop, {})
    items = []
    total_cost = 0.0
    for product, spec in FERTILISERS.items():
        nutrient = spec["nutrient"]
        band = envelope.get(nutrient)
        if not band:
            continue
        deficit = band["mean"] - getattr(site, nutrient)
        if deficit <= 1.0:
            continue
        kg_product = deficit / spec["fraction"]
        cost = kg_product * spec["price_per_kg"]
        total_cost += cost
        items.append({
            "product": product,
            "nutrient": nutrient,
            "deficit_kg_ha": round(deficit, 1),
            "quantity_kg_ha": round(kg_product, 1),
            "bags_50kg_per_ha": round(kg_product / 50.0, 1),
            "estimated_cost_inr": round(cost, 0),
        })
    return {
        "items": items,
        "total_cost_inr_per_ha": round(total_cost, 0),
        "note": "Indicative subsidised rates. Confirm against your Soil Health Card and local co-operative pricing.",
    }


def recommend(
    site: SiteConditions,
    top_n: int = 6,
    land_ha: float = 1.0,
    weights: dict[str, float] | None = None,
    state_id: str | None = None,
) -> dict[str, Any]:
    """Rank every crop for this site and return the top N with full reasoning.

    If `state_id` is given, the candidate set is first restricted to crops the
    state actually cultivates, and each crop's fitness is scaled by whether it
    is a major or minor crop there.
    """
    pipe = _bundle()["pipeline"]
    econ_all = _economics()
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    prior = regional_prior(state_id)

    proba = pipe.predict_proba(site.as_frame())[0]
    rf_proba = dict(zip(pipe.classes_, (float(p) for p in proba)))

    scored = []
    for crop, econ in econ_all.items():
        if prior is not None and crop not in prior:
            continue  # not cultivated in this state
        envelope_fit = _envelope_fit(crop, site)
        # The forest is decisive but spiky -- it concentrates almost all mass on
        # one class. The envelope gives a smooth signal across every crop, so we
        # lead with it and let the forest supply a confirmation bonus.
        agro = 0.75 * envelope_fit + 0.25 * rf_proba.get(crop, 0.0)
        prior_w = prior.get(crop, 1.0) if prior is not None else 1.0
        scored.append({
            "crop": crop, "econ": econ,
            "fitness": agro * prior_w,
            "agro_fit": agro,
            "prior": prior_w,
            "envelope_fit": envelope_fit,
            "rf_proba": rf_proba.get(crop, 0.0),
        })

    excluded_by_region = len(econ_all) - len(scored)
    viable = [s for s in scored if s["agro_fit"] >= FITNESS_GATE]
    gated_out = len(scored) - len(viable)
    if len(viable) < MIN_RESULTS:
        # Sparse candidate set (e.g. a cold-arid site like Ladakh). Show the
        # least-bad options rather than an empty screen, and let the low
        # confidence figures speak for themselves.
        viable = sorted(scored, key=lambda s: s["agro_fit"], reverse=True)[:MIN_RESULTS]

    annual_rain = site.annual_rain
    profits = [_annual_economics(s["crop"], s["econ"], state_id)["net_profit_per_ha_year"] for s in viable]
    # Rank on risk-adjusted expected return, not headline profit, and discount
    # by regional evidence as well as climate. A farmer cannot realise papaya's
    # national yield in a district with no papaya supply chain, buyer or
    # extension support -- so a headline figure the region cannot actually
    # deliver should not outrank a modest one it can.
    expected = [p * s["agro_fit"] * s["prior"] for p, s in zip(profits, viable)]
    # Net profit per hectare runs from about Rs 10k for a pulse to Rs 7 lakh for
    # papaya. Min-max normalising that raw range gives the single largest crop a
    # score of 1.0 and squashes everything else to near zero, so the ranking
    # collapses to "whichever crop has the biggest headline number". A log
    # transform keeps the ordering while restoring proportion between tiers.
    n_profit = _normalise([math.log1p(max(0.0, v)) for v in expected])

    results = []
    for i, s in enumerate(viable):
        crop, econ = s["crop"], s["econ"]
        water = _water_assessment(econ, annual_rain)
        risk_info = resolve_risk(crop, econ, state_id)
        risk = risk_info["risk"]
        score = (
            w["fitness"] * s["fitness"]
            + w["profit"] * n_profit[i]
            - w["water"] * water["dependence_ratio"]
            - w["risk"] * risk
        )
        economics = _annual_economics(crop, econ, state_id)
        results.append({
            "crop": crop,
            "display": econ["display"],
            "category": econ["category"],
            "season": econ["season"],
            "msp_backed": econ["msp_backed"],
            "score": round(float(score), 4),
            "fitness": round(float(s["fitness"]), 4),
            "fitness_pct": round(float(s["fitness"]) * 100, 1),
            "envelope_fit": round(float(s["envelope_fit"]), 4),
            "agro_fit_pct": round(float(s["agro_fit"]) * 100, 1),
            "regional_prior": s["prior"],
            "regional_tier": ("major" if s["prior"] == PRIOR_MAJOR
                              else "minor" if s["prior"] == PRIOR_MINOR else "unconstrained"),
            "model_confidence": round(float(s["rf_proba"]) * 100, 1),
            "risk_score": econ["risk_score"],
            "risk": {**risk_info, "risk": round(risk, 3)},
            "evidence": prior_source(state_id, crop),
            "area_share": state_stats(state_id).get(crop, {}).get("area_share"),
            "establishment_years": econ["establishment_years"],
            "economics": {
                **economics,
                "expected_profit_per_ha_year": round(
                    economics["net_profit_per_ha_year"] * s["agro_fit"], 0),
                "net_profit_total_inr": round(economics["net_profit_per_ha_year"] * land_ha, 0),
                "price_per_quintal": econ["price_per_quintal"],
                "yield_t_ha": econ["yield_t_ha"],
            },
            "water": water,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:top_n]
    for r in top:
        r["why"] = _explain(r["crop"], site)
        r["fertiliser_plan"] = _fertiliser_plan(r["crop"], site)

    for r in results:
        af = r["agro_fit_pct"]
        r["confidence"] = ("high" if af >= 60 else "moderate" if af >= 40
                           else "low" if af >= 25 else "marginal")

    best_fit = max(results, key=lambda r: r["fitness"])
    best_profit = max(results, key=lambda r: r["economics"]["net_profit_per_ha_year"])
    measured = sum(1 for r in results if r["economics"]["yield_source"] != "curated estimate")

    return {
        "site": {f: getattr(site, f) for f in FEATURES},
        "rainfall_annual_mm": round(annual_rain, 0),
        "land_ha": land_ha,
        "weights": w,
        "gate": {
            "threshold": FITNESS_GATE,
            "excluded_not_cultivated_in_region": excluded_by_region,
            "excluded_poor_agro_fit": gated_out,
            "crops_considered": len(viable),
            "regional_prior_applied": prior is not None,
            "crops_with_measured_yield": measured,
        },
        "recommendations": top,
        "headline": {
            "balanced_pick": top[0]["display"] if top else None,
            "confidence": top[0]["confidence"] if top else None,
            "best_agronomic_fit": best_fit["display"],
            "highest_profit": best_profit["display"],
            "profit_vs_fitness_differ": best_fit["crop"] != best_profit["crop"],
        },
        "all_scores": [
            {"crop": r["crop"], "display": r["display"], "score": r["score"],
             "fitness": r["fitness"],
             "net_profit_per_ha_year": r["economics"]["net_profit_per_ha_year"]}
            for r in results
        ],
    }


def recommend_for_state(
    state_id: str, top_n: int = 6, land_ha: float = 1.0,
    overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Recommendation driven by a state's agro-climatic profile.

    `overrides` lets a farmer replace any state average with their own Soil
    Health Card reading or a live sensor value.
    """
    state = get_state(state_id)
    if state is None:
        raise KeyError(f"Unknown state id: {state_id}")

    annual = float(state["rainfall_annual_mm"])
    values = {f: float(state[f]) for f in FEATURES if f != "rainfall"}
    values["rainfall"] = annual / MONTHS_PER_YEAR

    applied = {}
    for key, val in (overrides or {}).items():
        if val is None:
            continue
        if key == "rainfall_annual_mm":
            applied[key] = {"state_average": annual, "used": float(val)}
            annual = float(val)
            values["rainfall"] = annual / MONTHS_PER_YEAR
        elif key in values:
            applied[key] = {"state_average": values[key], "used": float(val)}
            values[key] = float(val)

    site = SiteConditions(**values, rainfall_annual_mm=annual)
    result = recommend(site, top_n=top_n, land_ha=land_ha, state_id=state["id"])
    result["state"] = {
        "id": state["id"], "name": state["name"], "zone": state["zone"],
        "soil": state["soil"], "lat": state["lat"], "lon": state["lon"],
        "rainfall_annual_mm": annual,
    }
    result["overrides_applied"] = applied
    return result
