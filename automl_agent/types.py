from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

TaskType = Literal["classification", "regression"]


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    target: str
    task_type: TaskType
    numeric_features: List[str]
    categorical_features: List[str]
    missing_values: Dict[str, int]
    target_summary: Dict[str, Any]


@dataclass
class DataBundle:
    dataset_name: str
    target: str
    task_type: TaskType
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    profile: DatasetProfile


@dataclass
class FeaturePlan:
    numeric_features: List[str]
    categorical_features: List[str]
    profile: DatasetProfile


@dataclass
class CandidateResult:
    name: str
    estimator: Any
    metrics: Dict[str, float]
    train_seconds: float
    # Mean cross-validation score in sklearn scorer convention (higher is better).
    cv_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class FeatureImportance:
    feature: str
    importance_mean: float
    importance_std: float


@dataclass
class ExplainabilityReport:
    method: str
    primary_metric: str
    importances: List[FeatureImportance]


@dataclass
class MonitoringBaseline:
    numeric: Dict[str, Dict[str, float]]
    categorical: Dict[str, Dict[str, Any]]
    drift_threshold_z: float


@dataclass
class PipelineReport:
    dataset: DatasetProfile
    leaderboard: List[CandidateResult]
    best_model_name: str
    best_metrics: Dict[str, float]
    tuned_metrics: Dict[str, float]
    explainability: Optional[ExplainabilityReport]
    monitoring_baseline: Optional[MonitoringBaseline]
    artifact_dir: Path
    model_bundle_path: Path
    best_cv_score: Optional[float] = None
    failed_candidates: List[Dict[str, str]] = field(default_factory=list)
    llm_summary: Optional[str] = None
    notes: List[str] = field(default_factory=list)
