from pathlib import Path

import joblib
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
            "model_version": "frontend-test",
            "model_name": "dummy",
            "pipeline": model,
            "target": "target",
            "task_type": "classification",
            "feature_columns": iris.data.columns.tolist(),
            "metrics": {"accuracy": 0.33},
            "profile": {},
            "explainability": None,
            "monitoring_baseline": None,
        },
        bundle_path,
    )
    return bundle_path


def test_frontend_index_and_assets_are_served(tmp_path: Path) -> None:
    from automl_agent.serving.app import create_app

    app = create_app(_bundle(tmp_path))
    with TestClient(app) as client:
        index = client.get("/")
        script = client.get("/static/app.js")
        styles = client.get("/static/styles.css")

    assert index.status_code == 200
    assert "AutoML Agent Console" in index.text
    assert script.status_code == 200
    assert "loadModel" in script.text
    assert styles.status_code == 200
    assert ".app-shell" in styles.text


def test_auth_status_is_public_when_auth_disabled(tmp_path: Path) -> None:
    from automl_agent.serving.app import create_app

    app = create_app(_bundle(tmp_path))
    with TestClient(app) as client:
        response = client.get("/auth/status")

    assert response.status_code == 200
    assert response.json() == {"enabled": False, "authenticated": False, "user": None}
