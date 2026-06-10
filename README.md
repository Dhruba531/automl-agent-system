# AutoML Agent System

A runnable multi-agent AutoML prototype inspired by **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (arXiv:2410.02958).

The system automates the tabular ML path from dataset retrieval to deployable FastAPI service:

- **Data Agent** retrieves built-in or CSV datasets, profiles them, infers task type, and creates train/test splits.
- **Feature Agent** builds preprocessing pipelines for numeric and categorical data.
- **Model Search Agent** trains candidate models in parallel.
- **Evaluation Agent** benchmarks candidates and selects the best model.
- **Hyperparameter Agent** tunes the winning model with Optuna, with a deterministic fallback if Optuna is unavailable.
- **Explainability Agent** computes permutation importance for the selected model.
- **Monitoring Agent** builds a training-data baseline for serving-time drift checks.
- **Deployment Agent** saves the model bundle and generates a FastAPI serving module.

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

## Paper-to-Code Agent

The Paper-to-Code Agent converts an academic paper into a runnable code project. It
drives the **Claude Code CLI in headless mode**, so generation is billed against your
signed-in Claude subscription rather than a metered `ANTHROPIC_API_KEY`.

Prerequisites: install [Claude Code](https://code.claude.com/docs) and sign in
(`claude` must be on your `PATH`).

Convert a local paper (PDF, Markdown, or text), an arXiv id/URL, or raw text:

```bash
# From a local file (PDF needs the optional 'paper' extra: pip install -e ".[paper]")
automl-agent paper2code path/to/paper.pdf --output artifacts/paper2code

# From an arXiv id (downloads the PDF)
automl-agent paper2code 2410.02958 --output artifacts/automl_agent_paper

# Choose a model and project name
automl-agent paper2code paper.md --model opus --name my_impl --output artifacts/my_impl
```

Useful flags:

- `--model` selects a Claude model alias or id (`opus`, `sonnet`, ...). Omit to use the subscription default.
- `--language` targets a language other than Python.
- `--max-chars` caps how much of the paper is sent (default 120k characters).
- `--overwrite` replaces existing files in the output directory.

Outputs in the project directory:

- The generated source tree (README, modules, an entry point, and tests).
- `paper2code_manifest.json` lists generated files, the paper source, and agent notes.
- `paper2code_raw_response.txt` preserves Claude's raw response for inspection.

The model is asked to emit files using an unambiguous delimiter format, so responses are
parsed deterministically and unsafe paths (absolute or `..`) are rejected.

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
