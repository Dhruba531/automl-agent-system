from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier

from automl_agent.serving.config import GoogleAuthSettings, ServingSettings
from automl_agent.serving.model_store import ModelBundleStore


def _bundle(path: Path) -> Path:
    iris = load_iris(as_frame=True)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(iris.data, iris.target)
    bundle_path = path / "model_bundle.joblib"
    joblib.dump(
        {
            "model_version": "test-version",
            "model_name": "dummy",
            "pipeline": model,
            "target": "target",
            "task_type": "classification",
            "feature_columns": iris.data.columns.tolist(),
            "metrics": {"accuracy": 0.33},
            "monitoring_baseline": {
                "numeric": {
                    column: {
                        "mean": float(iris.data[column].mean()),
                        "std": float(iris.data[column].std()),
                        "min": float(iris.data[column].min()),
                        "max": float(iris.data[column].max()),
                        "missing_rate": 0.0,
                    }
                    for column in iris.data.columns
                },
                "categorical": {},
                "drift_threshold_z": 3.0,
            },
        },
        bundle_path,
    )
    return bundle_path


@pytest.fixture(autouse=True)
def clean_auth_env(monkeypatch):
    for key in [
        "GOOGLE_AUTH_ENABLED",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "SESSION_SECRET_KEY",
        "GOOGLE_ALLOWED_DOMAINS",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_schema_is_open_when_google_auth_is_not_configured(tmp_path: Path) -> None:
    from automl_agent.serving.app import create_app

    app = create_app(_bundle(tmp_path))
    with TestClient(app) as client:
        response = client.get("/schema")

    assert response.status_code == 200
    assert response.json()["model_name"] == "dummy"
    assert response.json()["model_version"] == "test-version"


def test_schema_requires_session_when_google_auth_is_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_AUTH_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret-key-for-testing-only-32x")

    from automl_agent.serving.app import create_app

    app = create_app(_bundle(tmp_path))
    with TestClient(app) as client:
        response = client.get("/schema")

    assert response.status_code == 401
    assert response.json()["detail"] == "Google login required."


def test_enabled_google_auth_requires_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_AUTH_ENABLED", "true")
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)

    from automl_agent.serving.app import create_app

    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_ID"):
        create_app(Path("missing.joblib"))


def test_google_settings_parse_environment(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SESSION_SECRET_KEY", "session-secret-key-for-testing-only-32")
    monkeypatch.setenv("GOOGLE_ALLOWED_DOMAINS", "Example.com, team.example ")
    monkeypatch.setenv("SESSION_SECURE_COOKIES", "true")

    settings = GoogleAuthSettings.from_env()

    assert settings.enabled is True
    assert settings.allowed_domains == ["example.com", "team.example"]
    assert settings.secure_cookies is True
    settings.validate()


def test_serving_settings_can_be_injected(tmp_path: Path) -> None:
    from automl_agent.serving.app import create_app

    settings = ServingSettings(
        model_bundle_path=_bundle(tmp_path),
        google_auth=GoogleAuthSettings(
            client_id=None,
            client_secret=None,
            session_secret=None,
            enabled=False,
        ),
    )
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["bundle"] == str(settings.model_bundle_path)


def test_model_store_reports_missing_columns(tmp_path: Path) -> None:
    store = ModelBundleStore(_bundle(tmp_path))
    store.load()

    missing = store.missing_columns([{"sepal length (cm)": 5.1}])

    assert "sepal width (cm)" in missing


def test_model_store_checks_drift(tmp_path: Path) -> None:
    store = ModelBundleStore(_bundle(tmp_path))
    store.load()

    report = store.drift(
        [
            {
                "sepal length (cm)": 100.0,
                "sepal width (cm)": 100.0,
                "petal length (cm)": 100.0,
                "petal width (cm)": 100.0,
            }
        ]
    )

    assert report["monitoring_enabled"] is True
    assert report["drift_detected"] is True
    assert report["alerts"]
