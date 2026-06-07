from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_BUNDLE = Path(os.getenv("AUTOML_MODEL_BUNDLE", "artifacts/run/model_bundle.joblib"))


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    model_name: str
    task_type: str
    predictions: List[Any]
    probabilities: Optional[List[Any]] = None


def create_app(bundle_path: Path = DEFAULT_BUNDLE) -> FastAPI:
    app = FastAPI(title="AutoML Agent Model Server", version="0.1.0")
    state: Dict[str, Any] = {"bundle_path": bundle_path, "bundle": None}

    @app.on_event("startup")
    def load_bundle() -> None:
        if not state["bundle_path"].exists():
            raise RuntimeError(f"Model bundle not found: {state['bundle_path']}")
        state["bundle"] = joblib.load(state["bundle_path"])

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "bundle": str(state["bundle_path"])}

    @app.get("/schema")
    def schema() -> Dict[str, Any]:
        bundle = _bundle(state)
        return {
            "model_name": bundle["model_name"],
            "task_type": bundle["task_type"],
            "target": bundle["target"],
            "feature_columns": bundle["feature_columns"],
            "metrics": bundle["metrics"],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        bundle = _bundle(state)
        frame = pd.DataFrame(request.rows)
        missing = [column for column in bundle["feature_columns"] if column not in frame.columns]
        if missing:
            raise HTTPException(status_code=422, detail={"missing_columns": missing})
        frame = frame[bundle["feature_columns"]]
        pipeline = bundle["pipeline"]
        predictions = pipeline.predict(frame).tolist()
        probabilities = None
        if bundle["task_type"] == "classification" and hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(frame).tolist()
            except Exception:
                probabilities = None
        return PredictResponse(
            model_name=bundle["model_name"],
            task_type=bundle["task_type"],
            predictions=predictions,
            probabilities=probabilities,
        )

    return app


def _bundle(state: Dict[str, Any]) -> Dict[str, Any]:
    if state["bundle"] is None:
        if not state["bundle_path"].exists():
            raise HTTPException(status_code=503, detail=f"Model bundle not found: {state['bundle_path']}")
        state["bundle"] = joblib.load(state["bundle_path"])
    return state["bundle"]


app = create_app()
