# AutoML Agent System

A runnable multi-agent AutoML prototype inspired by **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (arXiv:2410.02958).

The system automates the tabular ML path from dataset retrieval to deployable FastAPI service:

- **Data Agent** retrieves built-in or CSV datasets, profiles them, infers task type, and creates train/test splits.
- **Feature Agent** builds preprocessing pipelines for numeric and categorical data.
- **Model Search Agent** trains candidate models in parallel.
- **Evaluation Agent** benchmarks candidates and selects the best model.
- **Hyperparameter Agent** tunes the winning model with Optuna, with a deterministic fallback if Optuna is unavailable.
- **Deployment Agent** saves the model bundle and generates a FastAPI serving module.

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

