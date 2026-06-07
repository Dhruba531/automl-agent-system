from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier


def _bundle(path: Path) -> Path:
    iris = load_iris(as_frame=True)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(iris.data, iris.target)
    bundle_path = path / "model_bundle.joblib"
    joblib.dump(
        {
            "model_name": "dummy",
            "pipeline": model,
            "target": "target",
            "task_type": "classification",
            "feature_columns": iris.data.columns.tolist(),
            "metrics": {"accuracy": 0.33},
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


def test_schema_requires_session_when_google_auth_is_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_AUTH_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret")

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
