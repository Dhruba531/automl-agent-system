from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from automl_agent.harness import ExperimentHarness, HarnessCase
from automl_agent.llm import RunPodConfig, RunPodConnector, VLLMConfig, VLLMConnector
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
    run.add_argument("--llm-base-url", help="vLLM OpenAI-compatible base URL (defaults to VLLM_BASE_URL).")
    run.add_argument("--llm-model", help="Model name to use (defaults to VLLM_MODEL/RUNPOD_MODEL or the first served model).")
    run.add_argument(
        "--runpod-endpoint-id",
        help="RunPod serverless vLLM endpoint id (defaults to RUNPOD_ENDPOINT_ID; requires RUNPOD_API_KEY).",
    )
    run.add_argument(
        "--prompt",
        help="Custom instruction to steer the LLM insight summary. "
        "Prefix with '@' to read the prompt from a file (e.g. --prompt @notes.txt).",
    )

    registry = subparsers.add_parser("registry", help="List model versions in a local registry.")
    registry.add_argument("--path", type=Path, default=Path("artifacts/registry.json"), help="Path to registry JSON.")

    harness = subparsers.add_parser("harness", help="Run repeatable experiment harness cases.")
    harness.add_argument("--config", type=Path, help="JSON harness config with a 'cases' array.")
    harness.add_argument("--output", type=Path, default=Path("artifacts/harness"), help="Harness output directory.")
    harness.add_argument("--dataset", action="append", help="Built-in dataset case. Can be repeated.")
    harness.add_argument("--workers", type=int, default=2, help="Default worker count for dataset cases.")
    harness.add_argument("--trials", type=int, default=0, help="Default tuning trials for dataset cases.")
    harness.add_argument("--fail-fast", action="store_true", help="Stop after the first failed case.")

    self_harness = subparsers.add_parser(
        "self-harness",
        help="Iteratively improve the AutoML harness from held-in/held-out evidence (Self-Harness).",
    )
    self_harness.add_argument(
        "--config", type=Path, required=True, help="JSON config with 'held_in' and 'held_out' case arrays."
    )
    self_harness.add_argument("--output", type=Path, default=Path("artifacts/self_harness"), help="Output directory.")
    self_harness.add_argument(
        "--memory",
        type=Path,
        help="Path to a memory JSON. Resumes from the stored harness and skips edits already tried; "
        "updated after the run. Improvement accumulates across runs.",
    )
    self_harness.add_argument(
        "--reset-memory", action="store_true", help="Ignore any existing memory and start fresh (still saved)."
    )
    self_harness.add_argument("--rounds", type=int, default=3, help="Number of improvement rounds (T).")
    self_harness.add_argument("--width", type=int, default=3, help="Parallel proposal width (K).")
    self_harness.add_argument("--workers", type=int, default=2, help="Workers per pipeline run.")
    self_harness.add_argument("--llm-base-url", help="vLLM base URL for the proposer (defaults to VLLM_BASE_URL).")
    self_harness.add_argument("--llm-model", help="Proposer model name (defaults to VLLM_MODEL/RUNPOD_MODEL).")
    self_harness.add_argument("--runpod-endpoint-id", help="RunPod endpoint id for the proposer (requires RUNPOD_API_KEY).")
    return parser


def _resolve_user_prompt(prompt: Optional[str]) -> Optional[str]:
    if not prompt:
        return None
    if prompt.startswith("@"):
        prompt_path = Path(prompt[1:])
        if not prompt_path.is_file():
            raise SystemExit(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    return prompt


def _build_llm_connector(args: argparse.Namespace) -> Optional[VLLMConnector]:
    config = VLLMConfig.from_env()
    if args.llm_base_url:
        config = config or VLLMConfig()
        config.base_url = args.llm_base_url
    if config:
        if args.llm_model:
            config.model = args.llm_model
        return VLLMConnector(config)

    runpod = RunPodConfig.from_env()
    if args.runpod_endpoint_id:
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise SystemExit("--runpod-endpoint-id requires the RUNPOD_API_KEY environment variable.")
        runpod = runpod or RunPodConfig(endpoint_id=args.runpod_endpoint_id, api_key=api_key)
        runpod.endpoint_id = args.runpod_endpoint_id
    if runpod:
        if args.llm_model:
            runpod.model = args.llm_model
        return RunPodConnector(runpod)
    return None


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        user_prompt = _resolve_user_prompt(args.prompt)
        connector = _build_llm_connector(args)
        orchestrator = AutoMLOrchestrator(max_workers=args.workers, tuning_trials=args.trials, llm_connector=connector)
        try:
            report = orchestrator.run(
                output_dir=args.output,
                dataset=args.dataset if not args.csv else None,
                csv_path=args.csv,
                target=args.target,
                task_type=args.task,
                user_prompt=user_prompt,
            )
        finally:
            if connector:
                connector.close()
        print(
            json.dumps(
                {
                    "best_model": report.best_model_name,
                    "best_metrics": report.best_metrics,
                    "tuned_metrics": report.tuned_metrics,
                    "model_bundle": str(report.model_bundle_path),
                    "report": str(report.artifact_dir / "pipeline_report.json"),
                    "llm_summary": str(report.artifact_dir / "llm_summary.md") if report.llm_summary else None,
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
    elif args.command == "self-harness":
        _run_self_harness(args)


def _run_self_harness(args: argparse.Namespace) -> None:
    from automl_agent.self_harness import HarnessCase as SelfHarnessCase
    from automl_agent.self_harness import HarnessMemory, SelfHarness

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    held_in = [SelfHarnessCase.from_dict(item) for item in payload.get("held_in", [])]
    held_out = [SelfHarnessCase.from_dict(item) for item in payload.get("held_out", [])]
    if not held_in or not held_out:
        raise SystemExit("Self-Harness config must define non-empty 'held_in' and 'held_out' arrays.")

    memory = None
    if args.memory is not None:
        memory = HarnessMemory(path=args.memory) if args.reset_memory else HarnessMemory.load(args.memory)

    connector = _build_llm_connector(args)
    loop = SelfHarness(
        held_in=held_in,
        held_out=held_out,
        output_dir=args.output,
        connector=connector,
        proposal_width=args.width,
        rounds=args.rounds,
        max_workers=args.workers,
    )
    try:
        result = loop.run(memory=memory)
    finally:
        if connector:
            connector.close()
    print(
        json.dumps(
            {
                "resumed_from_memory": result.resumed_from_memory,
                "held_in_pass": f"{result.initial_passed_in}/{result.total_in} -> "
                f"{result.final_passed_in}/{result.total_in}",
                "held_out_pass": f"{result.initial_passed_ho}/{result.total_ho} -> "
                f"{result.final_passed_ho}/{result.total_ho}",
                "final_config": result.final_config,
                "memory": str(args.memory) if memory is not None else None,
                "lineage": str(args.output / "lineage.json"),
                "summary": str(args.output / "summary.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
