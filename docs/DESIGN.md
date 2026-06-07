# Software Design Principles

This project is intentionally small, but it still follows production-oriented software engineering principles.

## Boundaries

- `automl_agent/agents/` contains pipeline agents with one clear responsibility each.
- `automl_agent/orchestrator.py` coordinates agents and owns workflow order.
- `automl_agent/serving/` owns API concerns only: configuration, authentication, request schemas, and model serving.
- `automl_agent/registry.py` owns append-only model version records.
- Artifacts are treated as runtime output and are excluded from source control.

## Principles Applied

- **Single Responsibility:** data loading, feature planning, model search, evaluation, tuning, deployment, auth, config, and serving are separated.
- **Open/Closed:** new agents, model candidates, auth providers, or serving stores can be added without rewriting the orchestrator contract.
- **Dependency Inversion:** `create_app` accepts explicit `ServingSettings` and `ModelBundleStore` instances for tests and alternate deployments.
- **DRY:** shared serving behavior is centralized in `ModelBundleStore`, `ServingSettings`, and Google auth helpers.
- **KISS:** the code favors standard Python, scikit-learn pipelines, FastAPI, and Authlib over custom frameworks.
- **YAGNI:** no distributed scheduler, database, or cloud-specific deployment layer is added until the use case needs it.
- **Least Privilege:** `/health` is public, while `/schema` and `/predict` require Google login when auth is enabled.
- **Fail Fast:** missing model bundles and incomplete auth configuration raise startup errors.
- **Testability:** auth behavior, config validation, and end-to-end pipeline behavior are covered with pytest.
- **Portability:** configuration comes from environment variables and the package supports Python 3.9+.
- **Observability:** packaged models include explainability and drift-monitoring artifacts so behavior can be inspected after deployment.

## Extension Points

- Add model families in `ModelSearchAgent._candidates`.
- Add tuning spaces in `HyperparameterAgent`.
- Add a new auth provider by creating another serving auth module and dependency.
- Add remote artifact loading by implementing a store with the same public methods as `ModelBundleStore`.
- Replace `ModelRegistry` with a database-backed registry while preserving the append-only record contract.
