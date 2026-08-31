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
PRIOR_MAJOR = 1.0
PRIOR_MINOR = 0.6

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


def regional_prior(state_id: str | None) -> dict[str, float] | None:
    """Prior weight per crop for a state; None means "no regional constraint".

    Returned as {crop: weight}. Crops absent from the map are not cultivated in
    that state and are excluded from the candidate set entirely.
    """
    if not state_id:
        return None
    entry = _calendar().get(state_id.upper())
    if entry is None:
        return None
    prior = {c: PRIOR_MAJOR for c in entry.get("major", [])}
    prior.update({c: PRIOR_MINOR for c in entry.get("minor", [])})
    return prior


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


def _annual_economics(crop: str, econ: dict[str, Any]) -> dict[str, float]:
    """Revenue and cost put on a common per-hectare-per-year footing.

    Comparing a 70-day mung bean crop against a 30-year mango orchard on raw
    per-cycle profit would be meaningless, so short crops are scaled by how many
    cycles fit in a year and perennials carry an amortised share of their
    establishment cost.
    """
    revenue_per_cycle = econ["yield_t_ha"] * 10.0 * econ["price_per_quintal"]
    opex = econ["opex_per_ha"]
    duration = max(econ["duration_days"], 1)

    if econ["establishment_years"] > 0 or duration >= 300:
        cycles_per_year = 1.0
    else:
        cycles_per_year = min(2.0, 365.0 / duration)

    gross = revenue_per_cycle * cycles_per_year
    running = opex * cycles_per_year

    capex = econ.get("capex_per_ha", 0)
    amortised_capex = capex / PRODUCTIVE_YEARS.get(crop, 15) if capex else 0.0

    net = gross - running - amortised_capex
    return {
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
    profits = [_annual_economics(s["crop"], s["econ"])["net_profit_per_ha_year"] for s in viable]
    # Rank on risk-adjusted expected return, not headline profit. A crop paying
    # Rs 6 lakh/ha that only half-fits the climate is not worth more than one
    # paying Rs 3 lakh that thrives -- weighting profit by agro-climatic fitness
    # stops high-value horticulture floating to the top of every single region.
    expected = [p * s["agro_fit"] for p, s in zip(profits, viable)]
    n_profit = _normalise(expected)

    results = []
    for i, s in enumerate(viable):
        crop, econ = s["crop"], s["econ"]
        water = _water_assessment(econ, annual_rain)
        risk = econ["risk_score"] / 5.0
        score = (
            w["fitness"] * s["fitness"]
            + w["profit"] * n_profit[i]
            - w["water"] * water["dependence_ratio"]
            - w["risk"] * risk
        )
        economics = _annual_economics(crop, econ)
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
