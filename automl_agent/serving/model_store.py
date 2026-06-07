from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from automl_agent.agents.monitoring import MonitoringAgent
from automl_agent.types import MonitoringBaseline


class ModelBundleStore:
    """Loads a packaged model once and exposes prediction-oriented operations."""

    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = bundle_path
        self._bundle: Optional[Dict[str, Any]] = None

    @property
    def bundle(self) -> Dict[str, Any]:
        if self._bundle is None:
            self.load()
        if self._bundle is None:
            raise RuntimeError("Model bundle did not load.")
        return self._bundle

    def load(self) -> None:
        if not self.bundle_path.exists():
            raise RuntimeError(f"Model bundle not found: {self.bundle_path}")
        self._bundle = joblib.load(self.bundle_path)

    def schema(self) -> Dict[str, Any]:
        bundle = self.bundle
        return {
            "model_version": bundle.get("model_version"),
            "model_name": bundle["model_name"],
            "task_type": bundle["task_type"],
            "target": bundle["target"],
            "feature_columns": bundle["feature_columns"],
            "metrics": bundle["metrics"],
        }

    def metadata(self) -> Dict[str, Any]:
        bundle = self.bundle
        return {
            "model_version": bundle.get("model_version"),
            "model_name": bundle["model_name"],
            "task_type": bundle["task_type"],
            "target": bundle["target"],
            "metrics": bundle["metrics"],
            "profile": bundle.get("profile"),
            "explainability": bundle.get("explainability"),
            "monitoring_enabled": bool(bundle.get("monitoring_baseline")),
        }

    def missing_columns(self, rows: List[Dict[str, Any]]) -> List[str]:
        frame = pd.DataFrame(rows)
        return [column for column in self.bundle["feature_columns"] if column not in frame.columns]

    def predict(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        bundle = self.bundle
        frame = pd.DataFrame(rows)
        frame = frame[bundle["feature_columns"]]
        pipeline = bundle["pipeline"]
        predictions = pipeline.predict(frame).tolist()
        probabilities = None
        if bundle["task_type"] == "classification" and hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(frame).tolist()
            except Exception:
                probabilities = None
        return {
            "model_name": bundle["model_name"],
            "task_type": bundle["task_type"],
            "predictions": predictions,
            "probabilities": probabilities,
        }

    def drift(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        baseline_payload = self.bundle.get("monitoring_baseline")
        if not baseline_payload:
            return {"monitoring_enabled": False, "drift_detected": False, "alerts": []}
        baseline = MonitoringBaseline(**baseline_payload)
        report = MonitoringAgent().check_drift(rows, baseline)
        report["monitoring_enabled"] = True
        return report
