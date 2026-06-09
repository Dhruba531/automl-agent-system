from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from automl_agent.harness import ExperimentHarness, HarnessCase
from automl_agent.orchestrator import AutoMLOrchestrator
from automl_agent.registry import ModelRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-agent AutoML pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run data, feature, model, tuning, evaluation, and deployment agents.")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--dataset", default="breast_cancer", help="Built-in dataset: iris, wine, breast_cancer, diabetes.")
    source.add_argument("--csv", type=Path, help="Path to a CSV dataset.")
    run.add_argument("--target", help="Target column. Defaults to the last CSV column.")
    run.add_argument("--task", choices=["classification", "regression"], help="Override inferred task type.")
    run.add_argument("--output", type=Path, default=Path("artifacts/run"), help="Artifact output directory.")
    run.add_argument("--workers", type=int, default=4, help="Parallel candidate training workers.")
    run.add_argument("--trials", type=int, default=20, help="Optuna tuning trials. Use 0 to skip tuning.")
    run.add_argument("--drift-threshold-z", type=float, default=3.0, dest="drift_threshold_z", help="Z-score threshold for drift detection.")
    run.add_argument("--models", help="Comma-separated model names to include, e.g. random_forest,ridge.")

    registry = subparsers.add_parser("registry", help="List model versions in a local registry.")
    registry.add_argument("--path", type=Path, default=Path("artifacts/registry.json"), help="Path to registry JSON.")

    harness = subparsers.add_parser("harness", help="Run repeatable experiment harness cases.")
    harness.add_argument("--config", type=Path, help="JSON harness config with a 'cases' array.")
    harness.add_argument("--output", type=Path, default=Path("artifacts/harness"), help="Harness output directory.")
    harness.add_argument("--dataset", action="append", help="Built-in dataset case. Can be repeated.")
    harness.add_argument("--workers", type=int, default=2, help="Default worker count for dataset cases.")
    harness.add_argument("--trials", type=int, default=0, help="Default tuning trials for dataset cases.")
    harness.add_argument("--fail-fast", action="store_true", help="Stop after the first failed case.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        include_models = set(args.models.split(",")) if args.models else None
        orchestrator = AutoMLOrchestrator(
            max_workers=args.workers,
            tuning_trials=args.trials,
            drift_threshold_z=args.drift_threshold_z,
            include_models=include_models,
        )
        report = orchestrator.run(
            output_dir=args.output,
            dataset=args.dataset if not args.csv else None,
            csv_path=args.csv,
            target=args.target,
            task_type=args.task,
        )
        print(
            json.dumps(
                {
                    "best_model": report.best_model_name,
                    "best_metrics": report.best_metrics,
                    "tuned_metrics": report.tuned_metrics,
                    "model_bundle": str(report.model_bundle_path),
                    "report": str(report.artifact_dir / "pipeline_report.json"),
                },
                indent=2,
            )
        )
    elif args.command == "registry":
        print(json.dumps(ModelRegistry(args.path).list(), indent=2))
    elif args.command == "harness":
        if args.config:
            harness, cases = ExperimentHarness.from_config_file(args.config, output_dir=args.output)
        else:
            datasets = args.dataset or ["iris", "diabetes"]
            cases = [
                HarnessCase(name=f"{dataset}-default", dataset=dataset, workers=args.workers, trials=args.trials)
                for dataset in datasets
            ]
            harness = ExperimentHarness(args.output)
        results = harness.run(cases, fail_fast=args.fail_fast)
        print(
            json.dumps(
                {
                    "output_dir": str(harness.output_dir),
                    "cases": len(results),
                    "passed": sum(1 for result in results if result.status == "passed"),
                    "failed": sum(1 for result in results if result.status == "failed"),
                    "results": str(harness.output_dir / "results.json"),
                    "summary": str(harness.output_dir / "summary.md"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
