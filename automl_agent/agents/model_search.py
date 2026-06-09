from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Set

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from automl_agent.agents.base import BaseAgent
from automl_agent.agents.evaluation import EvaluationAgent
from automl_agent.agents.feature import FeatureAgent
from automl_agent.types import CandidateResult, DataBundle, FeaturePlan


class ModelSearchAgent(BaseAgent):
    name = "Model Search Agent"

    def __init__(self, max_workers: int = 4, include_models: Optional[Set[str]] = None) -> None:
        super().__init__()
        self.max_workers = max_workers
        self.include_models = include_models
        self.evaluator = EvaluationAgent()

    def search(self, data: DataBundle, features: FeaturePlan) -> list[CandidateResult]:
        candidates = self._candidates(data.task_type)
        self.log(f"Training {len(candidates)} candidate models with up to {self.max_workers} workers.")
        results: list[CandidateResult] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._train_one, name, estimator, data, features): name
                for name, estimator in candidates.items()
            }
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _train_one(self, name: str, estimator, data: DataBundle, features: FeaturePlan) -> CandidateResult:
        start = time.perf_counter()
        try:
            preprocessor = FeatureAgent().build_preprocessor(features)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", clone(estimator))])
            pipeline.fit(data.X_train, data.y_train)
            result = self.evaluator.evaluate(name, pipeline, data)
            result.train_seconds = round(time.perf_counter() - start, 4)
            return result
        except Exception as exc:
            return CandidateResult(
                name=name,
                estimator=None,
                metrics={},
                train_seconds=round(time.perf_counter() - start, 4),
                error=str(exc),
            )

    def _candidates(self, task_type: str) -> Dict[str, object]:
        # n_jobs=1 because model-level parallelism is handled by ProcessPoolExecutor
        if task_type == "classification":
            all_candidates: Dict[str, object] = {
                "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
                "extra_trees": ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=1),
                "svc_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced"),
            }
        else:
            all_candidates = {
                "ridge": Ridge(),
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
                "extra_trees": ExtraTreesRegressor(n_estimators=120, random_state=42, n_jobs=1),
                "svr_rbf": SVR(kernel="rbf"),
            }
        if self.include_models:
            return {k: v for k, v in all_candidates.items() if k in self.include_models}
        return all_candidates
