from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib

from automl_agent.agents.base import BaseAgent
from automl_agent.registry import ModelRegistry
from automl_agent.types import CandidateResult, DataBundle, ExplainabilityReport, MonitoringBaseline


def _rel(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


class DeploymentAgent(BaseAgent):
    name = "Deployment Agent"

    def package(
        self,
        best: CandidateResult,
        data: DataBundle,
        artifact_dir: Path,
        explainability: Optional[ExplainabilityReport] = None,
        monitoring_baseline: Optional[MonitoringBaseline] = None,
    ) -> Dict[str, Path]:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bundle_path = artifact_dir / "model_bundle.joblib"
        profile_path = artifact_dir / "dataset_profile.json"
        report_path = artifact_dir / "metrics.json"
        manifest_path = artifact_dir / "manifest.json"
        explainability_path = artifact_dir / "explainability.json"
        monitoring_path = artifact_dir / "monitoring_baseline.json"

        bundle = {
            "model_version": model_version,
            "model_name": best.name,
            "pipeline": best.estimator,
            "target": data.target,
            "task_type": data.task_type,
            "feature_columns": data.X_train.columns.tolist(),
            "profile": asdict(data.profile),
            "metrics": best.metrics,
            "explainability": asdict(explainability) if explainability else None,
            "monitoring_baseline": asdict(monitoring_baseline) if monitoring_baseline else None,
        }
        joblib.dump(bundle, bundle_path)
        profile_path.write_text(json.dumps(asdict(data.profile), indent=2, default=str), encoding="utf-8")
        report_path.write_text(json.dumps({"model": best.name, "metrics": best.metrics}, indent=2), encoding="utf-8")
        if explainability:
            explainability_path.write_text(json.dumps(asdict(explainability), indent=2), encoding="utf-8")
        if monitoring_baseline:
            monitoring_path.write_text(json.dumps(asdict(monitoring_baseline), indent=2), encoding="utf-8")

        base = artifact_dir.parent
        manifest = {
            "model_version": model_version,
            "model_name": best.name,
            "task_type": data.task_type,
            "target": data.target,
            "metrics": best.metrics,
            "artifacts": {
                "bundle": _rel(bundle_path, base),
                "profile": _rel(profile_path, base),
                "metrics": _rel(report_path, base),
                "explainability": _rel(explainability_path, base) if explainability else None,
                "monitoring_baseline": _rel(monitoring_path, base) if monitoring_baseline else None,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        ModelRegistry(artifact_dir.parent / "registry.json").register(manifest)
        self.log(f"Packaged model bundle at {bundle_path}.")
        return {
            "bundle": bundle_path,
            "profile": profile_path,
            "metrics": report_path,
            "manifest": manifest_path,
        }
