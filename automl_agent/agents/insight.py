from __future__ import annotations

from typing import List, Optional

from automl_agent.agents.base import BaseAgent
from automl_agent.types import CandidateResult, DataBundle, ExplainabilityReport


class InsightAgent(BaseAgent):
    """Generates a natural-language run summary via an LLM connector.

    The connector only needs a ``chat(messages) -> str`` method, so any
    OpenAI-compatible backend (vLLM, Ollama, hosted APIs) can be plugged in.
    """

    name = "Insight Agent"

    def __init__(self, connector=None) -> None:
        super().__init__()
        self.connector = connector

    def summarize(
        self,
        data: DataBundle,
        leaderboard: List[CandidateResult],
        best: CandidateResult,
        explainability: Optional[ExplainabilityReport] = None,
        user_prompt: Optional[str] = None,
    ) -> Optional[str]:
        if self.connector is None:
            self.log("No LLM connector configured; skipping insight summary.")
            return None
        prompt = self._prompt(data, leaderboard, best, explainability, user_prompt)
        try:
            summary = self.connector.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an ML engineering assistant. Summarize AutoML runs in concise "
                            "markdown: performance assessment, model comparison, key feature drivers, "
                            "and recommended next steps."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            self.log(f"LLM summary failed ({exc}); continuing without it.")
            return None
        self.log("Generated LLM insight summary.")
        return summary

    def _prompt(
        self,
        data: DataBundle,
        leaderboard: List[CandidateResult],
        best: CandidateResult,
        explainability: Optional[ExplainabilityReport],
        user_prompt: Optional[str] = None,
    ) -> str:
        profile = data.profile
        lines = [
            f"Dataset: {data.dataset_name} ({profile.rows} rows, {profile.columns} columns, "
            f"task: {profile.task_type}, target: {profile.target})",
            "",
            "Leaderboard (cv_score uses sklearn scorer convention, higher is better):",
        ]
        for result in leaderboard:
            metrics = ", ".join(f"{key}={value:.4f}" for key, value in result.metrics.items())
            cv = "n/a" if result.cv_score is None else f"{result.cv_score:.4f}"
            lines.append(f"- {result.name}: cv_score={cv}; test metrics: {metrics}")
        lines.append("")
        lines.append(f"Selected model: {best.name}")
        if explainability and explainability.importances:
            lines.append("")
            lines.append("Top features by permutation importance:")
            for importance in explainability.importances[:8]:
                lines.append(f"- {importance.feature}: {importance.importance_mean:.4f}")
        if user_prompt and user_prompt.strip():
            lines.append("")
            lines.append("Additional instructions from the user (prioritize these):")
            lines.append(user_prompt.strip())
        return "\n".join(lines)
