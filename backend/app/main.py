"""FastAPI service for KrishiMitra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import chatbot
from .recommender import (
    FEATURES,
    SiteConditions,
    get_state,
    list_states,
    recommend,
    recommend_for_state,
)

app = FastAPI(
    title="KrishiMitra API",
    description="AI-based crop recommendation and scheme advisory for Indian farmers",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class SiteRequest(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Available nitrogen, kg/ha")
    P: float = Field(..., ge=0, le=200, description="Available phosphorus, kg/ha")
    K: float = Field(..., ge=0, le=250, description="Available potassium, kg/ha")
    temperature: float = Field(..., ge=-10, le=55, description="Mean temperature, °C")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity, %")
    ph: float = Field(..., ge=2, le=11, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall per growing cycle, mm")
    land_ha: float = Field(1.0, gt=0, le=1000)
    top_n: int = Field(6, ge=1, le=22)


class StateRequest(BaseModel):
    state_id: str
    land_ha: float = Field(1.0, gt=0, le=1000)
    top_n: int = Field(6, ge=1, le=22)
    overrides: dict[str, float] | None = Field(
        None,
        description="Override state averages with own Soil Health Card or sensor readings",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context_note: str | None = None
    history: list[dict[str, str]] | None = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    metrics_path = MODELS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return {
        "status": "ok",
        "model_trained": (MODELS_DIR / "crop_suitability.joblib").exists(),
        "chatbot_mode": "grok" if chatbot.GROK_KEY else "offline-retrieval",
        "metrics": metrics,
    }


@app.get("/api/states")
def states() -> dict[str, Any]:
    """Every state with its agro-climatic profile, for the map layer."""
    return {"states": list_states()}


@app.get("/api/states/{state_id}")
def state_detail(state_id: str) -> dict[str, Any]:
    state = get_state(state_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown state: {state_id}")
    return state


@app.post("/api/recommend/state")
def recommend_state(req: StateRequest) -> dict[str, Any]:
    """Recommendation for a state, optionally overridden with the farmer's own readings."""
    try:
        return recommend_for_state(
            req.state_id, top_n=req.top_n, land_ha=req.land_ha, overrides=req.overrides
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/recommend/custom")
def recommend_custom(req: SiteRequest) -> dict[str, Any]:
    """Recommendation from raw soil and climate readings, with no regional prior."""
    site = SiteConditions(**{f: getattr(req, f) for f in FEATURES})
    try:
        return recommend(site, top_n=req.top_n, land_ha=req.land_ha)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    """Top recommendation for every state, in one call.

    The map needs a fill colour per state on load. Doing that as 36 separate
    round trips is wasteful when the model is already resident in memory.
    """
    out = {}
    for state in list_states():
        try:
            result = recommend_for_state(state["id"], top_n=1)
        except (KeyError, FileNotFoundError):
            continue
        if not result["recommendations"]:
            continue
        top = result["recommendations"][0]
        out[state["id"]] = {
            "crop": top["crop"],
            "display": top["display"],
            "category": top["category"],
            "confidence": top["confidence"],
            "fitness_pct": top["agro_fit_pct"],
            "net": top["economics"]["net_profit_per_ha_year"],
            "expected": top["economics"]["expected_profit_per_ha_year"],
        }
    return {"top_by_state": out}


@app.get("/api/schemes")
def schemes() -> dict[str, Any]:
    return {"schemes": chatbot.list_schemes()}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    return chatbot.ask(req.message, context_note=req.context_note, history=req.history)
