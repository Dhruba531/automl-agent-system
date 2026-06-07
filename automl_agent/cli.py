from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from automl_agent.orchestrator import AutoMLOrchestrator


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
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        orchestrator = AutoMLOrchestrator(max_workers=args.workers, tuning_trials=args.trials)
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


if __name__ == "__main__":
    main()

