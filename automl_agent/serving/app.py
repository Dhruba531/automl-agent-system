from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from automl_agent.serving.auth import configure_google_auth, require_google_user
from automl_agent.serving.config import ServingSettings
from automl_agent.serving.model_store import ModelBundleStore
from automl_agent.serving.schemas import PredictRequest, PredictResponse


def create_app(
    bundle_path: Optional[Path] = None,
    settings: Optional[ServingSettings] = None,
    model_store: Optional[ModelBundleStore] = None,
) -> FastAPI:
    resolved_settings = settings or ServingSettings.from_env()
    if bundle_path is not None:
        resolved_settings = ServingSettings(bundle_path, resolved_settings.google_auth)
    resolved_settings.validate()
    store = model_store or ModelBundleStore(resolved_settings.model_bundle_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        store.load()
        yield

    app = FastAPI(title="AutoML Agent Model Server", version="0.1.0", lifespan=lifespan)
    configure_google_auth(app, resolved_settings.google_auth)
    app.state.model_store = store
    frontend_dir = files("automl_agent.frontend")
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "bundle": str(store.bundle_path)}

    @app.get("/schema")
    def schema(_user: Dict[str, Any] = Depends(require_google_user)) -> Dict[str, Any]:
        return store.schema()

    @app.get("/metadata")
    def metadata(_user: Dict[str, Any] = Depends(require_google_user)) -> Dict[str, Any]:
        return store.metadata()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest, _user: Dict[str, Any] = Depends(require_google_user)) -> PredictResponse:
        missing = store.missing_columns(request.rows)
        if missing:
            raise HTTPException(status_code=422, detail={"missing_columns": missing})
        return PredictResponse(**store.predict(request.rows))

    @app.post("/drift")
    def drift(request: PredictRequest, _user: Dict[str, Any] = Depends(require_google_user)) -> Dict[str, Any]:
        missing = store.missing_columns(request.rows)
        if missing:
            raise HTTPException(status_code=422, detail={"missing_columns": missing})
        return store.drift(request.rows)

    return app


app = create_app()
