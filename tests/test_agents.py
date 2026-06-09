from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import load_iris

from automl_agent.agents.feature import FeatureAgent
from automl_agent.agents.monitoring import MonitoringAgent
from automl_agent.registry import ModelRegistry
from automl_agent.serving.schemas import PredictRequest
from automl_agent.types import DataBundle, DatasetProfile, MonitoringBaseline


def _iris_bundle() -> DataBundle:
    iris = load_iris(as_frame=True)
    profile = DatasetProfile(
        rows=len(iris.data),
        columns=iris.data.shape[1] + 1,
        target="target",
        task_type="classification",
        numeric_features=iris.data.columns.tolist(),
        categorical_features=[],
        missing_values={},
        target_summary={},
    )
    return DataBundle(
        dataset_name="iris",
        target="target",
        task_type="classification",
        X_train=iris.data,
        X_test=iris.data,
        y_train=iris.target,
        y_test=iris.target,
        profile=profile,
    )


def test_feature_agent_plan_identifies_numeric_features() -> None:
    data = _iris_bundle()
    plan = FeatureAgent().plan(data)
    assert plan.numeric_features == load_iris(as_frame=True).data.columns.tolist()
    assert plan.categorical_features == []


def test_feature_agent_build_preprocessor_transforms_data() -> None:
    iris = load_iris(as_frame=True)
    data = _iris_bundle()
    agent = FeatureAgent()
    plan = agent.plan(data)
    preprocessor = agent.build_preprocessor(plan)
    transformed = preprocessor.fit_transform(iris.data)
    assert transformed.shape[0] == len(iris.data)
    assert transformed.shape[1] == len(iris.data.columns)


def test_feature_agent_plan_and_build_returns_both() -> None:
    data = _iris_bundle()
    agent = FeatureAgent()
    plan, preprocessor = agent.plan_and_build(data)
    assert plan.numeric_features
    transformed = preprocessor.fit_transform(load_iris(as_frame=True).data)
    assert transformed.shape[0] > 0


def test_feature_agent_handles_categorical_columns() -> None:
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    profile = DatasetProfile(
        rows=3, columns=3, target="label", task_type="classification",
        numeric_features=["num"], categorical_features=["cat"],
        missing_values={}, target_summary={},
    )
    data = DataBundle(
        dataset_name="test", target="label", task_type="classification",
        X_train=df, X_test=df,
        y_train=pd.Series([0, 1, 0]), y_test=pd.Series([0, 1, 0]),
        profile=profile,
    )
    agent = FeatureAgent()
    plan = agent.plan(data)
    preprocessor = agent.build_preprocessor(plan)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3


def test_monitoring_agent_detects_numeric_drift() -> None:
    baseline = MonitoringBaseline(
        numeric={"val": {"mean": 5.0, "std": 1.0, "min": 2.0, "max": 8.0, "missing_rate": 0.0}},
        categorical={},
        drift_threshold_z=3.0,
    )
    report = MonitoringAgent().check_drift([{"val": 100.0}], baseline)
    assert report["drift_detected"] is True
    assert any(a["reason"] == "mean_shift" for a in report["alerts"])


def test_monitoring_agent_no_drift_within_threshold() -> None:
    baseline = MonitoringBaseline(
        numeric={"val": {"mean": 5.0, "std": 1.0, "min": 2.0, "max": 8.0, "missing_rate": 0.0}},
        categorical={},
        drift_threshold_z=3.0,
    )
    report = MonitoringAgent().check_drift([{"val": 5.5}], baseline)
    assert report["drift_detected"] is False
    assert report["alerts"] == []


def test_monitoring_agent_detects_categorical_drift() -> None:
    baseline = MonitoringBaseline(
        numeric={},
        categorical={"color": {"top_values": {"red": 0.6, "blue": 0.4}, "missing_rate": 0.0}},
        drift_threshold_z=3.0,
    )
    report = MonitoringAgent().check_drift([{"color": "green"}, {"color": "purple"}], baseline)
    assert report["drift_detected"] is True
    unseen = next(a for a in report["alerts"] if a["reason"] == "unseen_categories")
    assert set(unseen["values"]) & {"green", "purple"}


def test_monitoring_agent_no_drift_known_categories() -> None:
    baseline = MonitoringBaseline(
        numeric={},
        categorical={"color": {"top_values": {"red": 0.6, "blue": 0.4}, "missing_rate": 0.0}},
        drift_threshold_z=3.0,
    )
    report = MonitoringAgent().check_drift([{"color": "red"}, {"color": "blue"}], baseline)
    assert report["drift_detected"] is False


def test_registry_appends_entries(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    first = registry.register({"model": "A", "version": "1"})
    second = registry.register({"model": "B", "version": "2"})
    assert len(first) == 1
    assert len(second) == 2
    assert registry.list()[0]["model"] == "A"
    assert registry.list()[1]["model"] == "B"


def test_registry_creates_file_and_parent_dirs(tmp_path) -> None:
    path = tmp_path / "nested" / "registry.json"
    registry = ModelRegistry(path)
    assert not path.exists()
    registry.register({"model": "X"})
    assert path.exists()
    assert len(registry.list()) == 1


def test_registry_prunes_to_max_entries(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    for i in range(5):
        registry.register({"model": str(i)}, max_entries=3)
    entries = registry.list()
    assert len(entries) == 3
    assert entries[0]["model"] == "2"
    assert entries[-1]["model"] == "4"


def test_predict_request_rejects_empty_rows() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(rows=[])


def test_predict_request_rejects_oversized_batch() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(rows=[{"x": 1}] * 1001)


def test_predict_request_accepts_valid_rows() -> None:
    req = PredictRequest(rows=[{"a": 1, "b": 2}])
    assert len(req.rows) == 1


def test_predict_request_accepts_max_batch() -> None:
    req = PredictRequest(rows=[{"x": i} for i in range(1000)])
    assert len(req.rows) == 1000
