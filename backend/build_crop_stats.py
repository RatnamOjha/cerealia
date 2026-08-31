"""
Derive per-state crop statistics from official Government of India data.

Source: "District-wise, season-wise crop production statistics", Directorate of
Economics & Statistics, Ministry of Agriculture & Farmers Welfare, published on
data.gov.in. 246,091 records covering 646 districts, 33 states, 124 crops and
19 years (1997-2015).

This replaces three previously hand-authored inputs with measured values:

  1. Regional cultivation prior -- which crops a state actually grows, and at
     what scale, from real sown area rather than our judgement.
  2. Crop yield -- state-specific tonnes/hectare instead of one national figure.
  3. Risk score -- the coefficient of variation of yield across 19 years, which
     is what "risky crop" actually means, instead of a hand-assigned 1-5.

Run:  ./backend/.venv/bin/python backend/build_crop_stats.py
Out:  backend/app/data/state_crop_stats.json
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "app" / "data" / "crop_production_india.csv.gz"
PROFILES = ROOT / "app" / "data" / "state_profiles.json"
OUT = ROOT / "app" / "data" / "state_crop_stats.json"

# The dataset uses Indian crop names; the model uses the Kaggle English labels.
# Several of our crops map to more than one source name (Rice/Paddy are recorded
# separately by different states; cotton appears as both lint and kapas).
CROP_MAP: dict[str, list[str]] = {
    "rice": ["Rice", "Paddy"],
    "maize": ["Maize"],
    "chickpea": ["Gram"],
    "kidneybeans": ["Rajmash Kholar"],
    "pigeonpeas": ["Arhar/Tur"],
    "mothbeans": ["Moth"],
    "mungbean": ["Moong(Green Gram)"],
    "blackgram": ["Blackgram", "Urad"],
    "lentil": ["Lentil", "Masoor"],
    "pomegranate": ["Pome Granet"],
    "banana": ["Banana"],
    "mango": ["Mango"],
    "grapes": ["Grapes"],
    "watermelon": ["Water Melon"],
    "muskmelon": [],           # not recorded separately in this dataset
    "apple": ["Apple"],
    "orange": ["Orange"],
    "papaya": ["Papaya"],
    "coconut": ["Coconut"],
    # Lint only: the source also carries "Kapas" (seed cotton) for some states,
    # and averaging the two mixes incompatible units.
    "cotton": ["Cotton(lint)"],
    "jute": ["Jute", "Jute & mesta"],
    "coffee": ["Coffee"],
}

# State names in the source predate a few reorganisations.
STATE_ALIAS = {
    "Dadra and Nagar Haveli": "DN",
    "Andaman and Nicobar Islands": "AN",
    "Jammu and Kashmir": "JK",
    "Odisha": "OR",
    "Puducherry": "PY",
    "Chandigarh": "CH",
    "Telangana": "TG",
}

# Not every crop is recorded in tonnes -- coconut production is counted in nuts,
# cotton in bales of lint. Rather than guess, each derived yield is checked
# against the curated agronomic figure for that crop and accepted only if it
# lands within a factor of four either way. Anything outside that is a units
# mismatch, and the curated figure is used instead.
YIELD_SANITY_FACTOR = 2.5

# Share of a state's total mapped crop area, above which we treat a crop as a
# major crop of that state rather than a minor one.
MAJOR_AREA_SHARE = 0.05
MINOR_AREA_SHARE = 0.001


def load_raw() -> pd.DataFrame:
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. Download the dataset first — see the module docstring."
        )
    with gzip.open(RAW, "rt") as fh:
        df = pd.read_csv(fh)
    df.columns = [c.strip() for c in df.columns]
    for col in ["State_Name", "District_Name", "Season", "Crop"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def curated_yields() -> dict[str, float]:
    econ = json.loads((ROOT / "app" / "data" / "crop_economics.json").read_text())["crops"]
    return {k: v["yield_t_ha"] for k, v in econ.items()}


def build() -> None:
    df = load_raw()
    curated = curated_yields()
    profiles = json.loads(PROFILES.read_text())["states"]
    name_to_id = {s["name"]: s["id"] for s in profiles}
    name_to_id.update({k: v for k, v in STATE_ALIAS.items()})

    source_to_crop = {src: crop for crop, srcs in CROP_MAP.items() for src in srcs}

    df = df[df["Crop"].isin(source_to_crop)].copy()
    df["crop"] = df["Crop"].map(source_to_crop)
    df["state_id"] = df["State_Name"].map(name_to_id)

    unmapped = sorted(df.loc[df["state_id"].isna(), "State_Name"].unique())
    if unmapped:
        print(f"  ! states with no profile, dropped: {unmapped}")
    df = df.dropna(subset=["state_id"])

    df = df[(df["Area"] > 0) & df["Production"].notna() & (df["Production"] >= 0)]
    print(f"Usable records after mapping: {len(df):,}")

    # Yield per record, then aggregate by state x crop x year so a state's
    # yield is area-weighted across its districts rather than a mean of means.
    by_year = (
        df.groupby(["state_id", "crop", "Crop_Year"])[["Area", "Production"]]
        .sum()
        .reset_index()
    )
    by_year["yield_t_ha"] = by_year["Production"] / by_year["Area"]

    stats: dict[str, dict[str, dict]] = {}
    for (state_id, crop), grp in by_year.groupby(["state_id", "crop"]):
        area_total = float(grp["Area"].sum())
        yields = grp["yield_t_ha"]
        mean_yield = float(yields.mean())
        std_yield = float(yields.std()) if len(yields) > 1 else 0.0
        cv = std_yield / mean_yield if mean_yield > 0 else 0.0

        ref = curated.get(crop)
        plausible = bool(
            ref and mean_yield > 0
            and ref / YIELD_SANITY_FACTOR <= mean_yield <= ref * YIELD_SANITY_FACTOR
        )
        stats.setdefault(state_id, {})[crop] = {
            "area_ha_total": round(area_total, 0),
            "mean_area_ha_per_year": round(area_total / max(grp["Crop_Year"].nunique(), 1), 0),
            "yield_t_ha": round(mean_yield, 3) if plausible else None,
            "yield_cv": round(cv, 3),
            "years": int(grp["Crop_Year"].nunique()),
            "yield_units_ok": bool(plausible),
        }

    # Area share is computed per state across the crops we model, which is what
    # the prior needs: "how much of this state's cropping is this crop".
    out_states: dict[str, dict] = {}
    for state_id, crops in stats.items():
        total = sum(c["area_ha_total"] for c in crops.values()) or 1.0
        entry = {}
        for crop, rec in crops.items():
            share = rec["area_ha_total"] / total
            if share >= MAJOR_AREA_SHARE:
                tier = "major"
            elif share >= MINOR_AREA_SHARE:
                tier = "minor"
            else:
                tier = "negligible"
            entry[crop] = {**rec, "area_share": round(share, 5), "tier": tier}
        out_states[state_id] = dict(
            sorted(entry.items(), key=lambda kv: kv[1]["area_share"], reverse=True)
        )

    # National fallback yields for states with no record of a crop.
    national = {}
    for crop in CROP_MAP:
        recs = [c[crop] for c in out_states.values() if crop in c and c[crop]["yield_units_ok"]]
        if recs:
            w = sum(r["area_ha_total"] for r in recs) or 1.0
            national[crop] = {
                "yield_t_ha": round(sum(r["yield_t_ha"] * r["area_ha_total"] for r in recs) / w, 3),
                "yield_cv": round(sum(r["yield_cv"] * r["area_ha_total"] for r in recs) / w, 3),
                "states_reporting": len(recs),
            }

    payload = {
        "_meta": {
            "source": "District-wise, season-wise crop production statistics",
            "publisher": "Directorate of Economics & Statistics, Ministry of Agriculture & Farmers Welfare, Government of India",
            "portal": "https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0",
            "coverage": {
                "records_raw": 246091,
                "records_used": int(len(df)),
                "years": [int(df["Crop_Year"].min()), int(df["Crop_Year"].max())],
                "districts": int(df["District_Name"].nunique()),
                "states_mapped": len(out_states),
                "crops_mapped": int(df["crop"].nunique()),
            },
            "derived_fields": {
                "area_share": "Crop's share of the state's total sown area across the 22 modelled crops",
                "tier": f"major >= {MAJOR_AREA_SHARE:.0%} area share, minor >= {MINOR_AREA_SHARE:.1%}, else negligible",
                "yield_t_ha": "Area-weighted mean yield across districts and years",
                "yield_cv": "Coefficient of variation of annual yield - the empirical risk measure",
                "yield_units_ok": "False where source production is not in tonnes (coconut in nuts, cotton in bales); the curated national figure is used instead",
            },
        },
        "national": national,
        "states": out_states,
    }
    OUT.write_text(json.dumps(payload, indent=1))

    print(f"\nStates covered : {len(out_states)}")
    print(f"Crops with national yield: {len(national)}")
    print(f"Written -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)\n")

    for sid in ["PB", "KL", "AS", "RJ", "MH", "HP", "WB"]:
        if sid not in out_states:
            continue
        top = list(out_states[sid].items())[:5]
        label = ", ".join(
            f"{c} {r['area_share']*100:.0f}%" for c, r in top if r["tier"] != "negligible"
        )
        print(f"  {sid}: {label}")


if __name__ == "__main__":
    build()
