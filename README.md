# AutoML Agent System

A runnable multi-agent AutoML prototype inspired by **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (arXiv:2410.02958).

The system automates the tabular ML path from dataset retrieval to deployable FastAPI service:

- **Data Agent** retrieves built-in or CSV datasets, drops duplicate rows, missing-target rows, and constant or ID-like columns, profiles the result, infers task type, and creates train/test splits.
- **Feature Agent** builds preprocessing pipelines for numeric and categorical data, capping one-hot cardinality so high-cardinality columns cannot explode the feature space.
- **Model Search Agent** trains candidate models in parallel (including histogram gradient boosting) and scores each with cross-validation on the training split.
- **Evaluation Agent** ranks candidates by cross-validated score, keeping the test split as a true holdout for reporting (with multiclass-aware ROC AUC).
- **Hyperparameter Agent** tunes the winning model with Optuna, with a deterministic fallback if Optuna is unavailable; the tuned model replaces the original only if its cross-validated score is better.
- **Explainability Agent** computes permutation importance for the selected model.
- **Monitoring Agent** builds a training-data baseline for serving-time drift checks.
- **Deployment Agent** saves the model bundle and generates a FastAPI serving module.
- **Insight Agent** (optional) summarizes the run in natural language through a vLLM connector.

The architecture notes in [`docs/DESIGN.md`](docs/DESIGN.md) describe the software engineering principles used across the project.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run an end-to-end pipeline on a built-in dataset:

```bash
automl-agent run --dataset breast_cancer --output artifacts/breast_cancer
```

Or run directly with Python:

```bash
python -m automl_agent.cli run --dataset iris --output artifacts/iris
```

Serve the best model:

```bash
uvicorn automl_agent.serving.app:app --host 127.0.0.1 --port 8000
```

Open the frontend console:

```text
http://127.0.0.1:8000/
```

The bundled UI provides model metadata, metrics, schema-aware prediction input, drift checks, explainability, and Google sign-in status.

Set `AUTOML_MODEL_BUNDLE` when serving a specific artifact:

```bash
AUTOML_MODEL_BUNDLE=artifacts/breast_cancer/model_bundle.joblib uvicorn automl_agent.serving.app:app
```

Predict:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"rows":[{"mean radius":14.0,"mean texture":20.0,"mean perimeter":90.0,"mean area":600.0}]}'
```

For realistic predictions, send all columns listed by `GET /schema`.

## Google Authentication

The FastAPI serving app supports Google OpenID Connect login. Authentication is enabled automatically when `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are present, or explicitly with `GOOGLE_AUTH_ENABLED=true`.

Create an OAuth client in Google Cloud Console and add this redirect URI for local development:

```text
http://127.0.0.1:8000/auth/callback
```

Then run the server with credentials:

```bash
export GOOGLE_AUTH_ENABLED=true
export GOOGLE_CLIENT_ID="your-google-oauth-client-id"
export GOOGLE_CLIENT_SECRET="your-google-oauth-client-secret"
export SESSION_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export AUTOML_MODEL_BUNDLE=artifacts/breast_cancer/model_bundle.joblib

uvicorn automl_agent.serving.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/auth/login` to sign in. After login, `/schema`, `/predict`, and `/auth/me` are available to the authenticated browser session. `/health` remains public.

Optional settings:

- `GOOGLE_ALLOWED_DOMAINS=example.com,team.example` restricts access to Google accounts from specific email or hosted domains.
- `GOOGLE_REDIRECT_URI=https://your-domain.com/auth/callback` overrides callback URL generation behind a proxy.
- `AUTH_SUCCESS_REDIRECT=/schema` controls where users land after login.
- `SESSION_SECURE_COOKIES=true` should be used behind HTTPS.

## Model Lifecycle Features

Each pipeline run now creates a richer artifact set:

- `model_bundle.joblib` includes the trained pipeline, metrics, model version, explainability report, and monitoring baseline.
- `manifest.json` records artifact paths, task type, target, metrics, and model version.
- `explainability.json` contains permutation-importance results.
- `monitoring_baseline.json` stores baseline feature statistics for drift checks.
- `registry.json` is an append-only local model registry in the parent artifacts directory.

List registered model versions:

```bash
automl-agent registry --path artifacts/registry.json
```

Serving endpoints:

- `GET /` serves the frontend console.
- `GET /metadata` returns profile, metrics, explainability, and monitoring availability.
- `POST /drift` checks incoming rows against the training baseline.
- `POST /predict` still returns model predictions and class probabilities when available.

## Experiment Harness

The harness runs repeatable benchmark cases and writes aggregate outputs:

```bash
automl-agent harness --config examples/harness.json --output artifacts/harness
```

Or run built-in dataset cases directly:

```bash
automl-agent harness --dataset iris --dataset diabetes --workers 2 --trials 0
```

Harness outputs:

- `results.json` contains structured per-case results.
- `results.csv` is spreadsheet-friendly.
- `summary.md` is a readable leaderboard.
- Each case gets its own full AutoML artifact directory.

## vLLM Connector

The pipeline can generate a natural-language run summary through any [vLLM](https://docs.vllm.ai) server exposing the OpenAI-compatible API. Start a server, for example:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

Then point the pipeline at it:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
automl-agent run --dataset breast_cancer --output artifacts/breast_cancer
```

When `VLLM_BASE_URL` is set, the Insight Agent sends the dataset profile, leaderboard, and top features to the model and writes `llm_summary.md` next to the other artifacts. The summary also appears in `pipeline_report.json`. LLM failures never fail the pipeline; the run continues without a summary.

Configuration:

- `VLLM_BASE_URL` (or `--llm-base-url`) enables the connector.
- `VLLM_MODEL` (or `--llm-model`) selects a model; defaults to the first model the server lists.
- `VLLM_API_KEY` sends a bearer token when the server requires one.
- `VLLM_MAX_TOKENS`, `VLLM_TEMPERATURE`, `VLLM_TIMEOUT_SECONDS` tune the request.

Programmatic use:

```python
from automl_agent.llm import VLLMConfig, VLLMConnector
from automl_agent.orchestrator import AutoMLOrchestrator

connector = VLLMConnector(VLLMConfig(base_url="http://localhost:8000/v1"))
orchestrator = AutoMLOrchestrator(llm_connector=connector)
```

Any object with a `chat(messages) -> str` method works as a connector, so other OpenAI-compatible backends can be swapped in.

### Custom Prompt

Steer the summary with `--prompt` to focus the model on what matters to you:

```bash
automl-agent run --dataset breast_cancer \
  --prompt "Explain the result for a non-technical stakeholder and flag deployment risks."
```

The instruction is appended to the run context the Insight Agent sends to the model, so the LLM keeps the leaderboard and feature facts but follows your steer. Pass `--prompt @path/to/prompt.txt` to read a longer prompt from a file. The same text can be supplied programmatically via `AutoMLOrchestrator.run(..., user_prompt=...)`.

## RunPod Connector

If you don't have a local GPU, the same insight summaries can run on [RunPod](https://docs.runpod.io) serverless GPU workers. Deploy a serverless vLLM endpoint from the RunPod console, then:

```bash
export RUNPOD_ENDPOINT_ID=your-endpoint-id
export RUNPOD_API_KEY=your-runpod-api-key
automl-agent run --dataset breast_cancer --output artifacts/breast_cancer
```

The connector talks to the endpoint's OpenAI-compatible route (`https://api.runpod.ai/v2/<endpoint_id>/openai/v1`) and otherwise behaves exactly like the vLLM connector, including model auto-discovery and writing `llm_summary.md`.

Configuration:

- `RUNPOD_ENDPOINT_ID` (or `--runpod-endpoint-id`) selects the serverless endpoint.
- `RUNPOD_API_KEY` authenticates; it is always read from the environment, never from flags.
- `RUNPOD_MODEL` (or `--llm-model`) selects a model; defaults to the first model the worker lists.
- `RUNPOD_MAX_TOKENS`, `RUNPOD_TEMPERATURE`, `RUNPOD_TIMEOUT_SECONDS` tune the request (the timeout defaults to 120s to absorb cold starts).

When both are configured, a local `VLLM_BASE_URL` takes priority over RunPod.

Programmatic use:

```python
from automl_agent.llm import RunPodConfig, RunPodConnector
from automl_agent.orchestrator import AutoMLOrchestrator

connector = RunPodConnector(RunPodConfig(endpoint_id="your-endpoint-id", api_key="..."))
orchestrator = AutoMLOrchestrator(llm_connector=connector)
```

## CSV Usage

```bash
automl-agent run --csv path/to/data.csv --target target_column --output artifacts/custom
```

If `--task` is omitted, the Data Agent infers `classification` or `regression`.

## Development

```bash
pytest
```

The implementation is intentionally small and readable so it can become a research scaffold: add agent memory, LLM planning, richer data retrieval, distributed training, model cards, drift monitoring, or cloud deployment without changing the core contracts.
