from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from automl_agent.agents.base import BaseAgent
from automl_agent.types import DataBundle, MonitoringBaseline


class MonitoringAgent(BaseAgent):
    name = "Monitoring Agent"

    def build_baseline(self, data: DataBundle, drift_threshold_z: float = 3.0) -> MonitoringBaseline:
        numeric = {
            column: self._numeric_stats(data.X_train[column])
            for column in data.profile.numeric_features
            if column in data.X_train
        }
        categorical = {
            column: self._categorical_stats(data.X_train[column])
            for column in data.profile.categorical_features
            if column in data.X_train
        }
        self.log(f"Created monitoring baseline for {len(numeric)} numeric and {len(categorical)} categorical features.")
        return MonitoringBaseline(numeric=numeric, categorical=categorical, drift_threshold_z=drift_threshold_z)

    def check_drift(self, rows: List[Dict[str, Any]], baseline: MonitoringBaseline) -> Dict[str, Any]:
        frame = pd.DataFrame(rows)
        numeric_checks = {}
        alerts = []
        for column, stats in baseline.numeric.items():
            if column not in frame:
                alerts.append({"feature": column, "reason": "missing_feature"})
                continue
            observed = pd.to_numeric(frame[column], errors="coerce")
            observed_mean = float(observed.mean()) if not observed.dropna().empty else None
            if observed_mean is None:
                z_shift = None
            else:
                z_shift = abs(observed_mean - stats["mean"]) / max(stats["std"], 1e-9)
            drifted = z_shift is not None and z_shift > baseline.drift_threshold_z
            if drifted:
                alerts.append({"feature": column, "reason": "mean_shift", "z_shift": z_shift})
            numeric_checks[column] = {
                "baseline_mean": stats["mean"],
                "observed_mean": observed_mean,
                "z_shift": z_shift,
                "drifted": drifted,
                "missing_rate": float(observed.isna().mean()),
            }
        categorical_checks = {}
        for column, stats in baseline.categorical.items():
            if column not in frame:
                alerts.append({"feature": column, "reason": "missing_feature"})
                continue
            observed_top = frame[column].astype(str).value_counts(normalize=True).head(5).to_dict()
            unseen_values = sorted(set(observed_top) - set(stats["top_values"]))
            if unseen_values:
                alerts.append({"feature": column, "reason": "unseen_categories", "values": unseen_values[:5]})
            categorical_checks[column] = {"observed_top_values": observed_top, "unseen_values": unseen_values}
        return {
            "drift_detected": bool(alerts),
            "threshold_z": baseline.drift_threshold_z,
            "alerts": alerts,
            "numeric": numeric_checks,
            "categorical": categorical_checks,
        }

    def _numeric_stats(self, series: pd.Series) -> Dict[str, float]:
        values = pd.to_numeric(series, errors="coerce")
        return {
            "mean": float(values.mean()),
            "std": float(values.std() or 0.0),
            "min": float(values.min()),
            "max": float(values.max()),
            "missing_rate": float(values.isna().mean()),
        }

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        return {
            "top_values": series.astype(str).value_counts(normalize=True).head(20).to_dict(),
            "missing_rate": float(series.isna().mean()),
        }

