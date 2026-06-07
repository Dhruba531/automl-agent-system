from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import joblib

from automl_agent.agents.base import BaseAgent
from automl_agent.types import CandidateResult, DataBundle


class DeploymentAgent(BaseAgent):
    name = "Deployment Agent"

    def package(self, best: CandidateResult, data: DataBundle, artifact_dir: Path) -> Dict[str, Path]:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = artifact_dir / "model_bundle.joblib"
        profile_path = artifact_dir / "dataset_profile.json"
        report_path = artifact_dir / "metrics.json"

        bundle = {
            "model_name": best.name,
            "pipeline": best.estimator,
            "target": data.target,
            "task_type": data.task_type,
            "feature_columns": data.X_train.columns.tolist(),
            "profile": asdict(data.profile),
            "metrics": best.metrics,
        }
        joblib.dump(bundle, bundle_path)
        profile_path.write_text(json.dumps(asdict(data.profile), indent=2, default=str), encoding="utf-8")
        report_path.write_text(json.dumps({"model": best.name, "metrics": best.metrics}, indent=2), encoding="utf-8")
        self.log(f"Packaged model bundle at {bundle_path}.")
        return {"bundle": bundle_path, "profile": profile_path, "metrics": report_path}

