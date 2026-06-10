from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from automl_agent.agents.base import BaseAgent
from automl_agent.agents.evaluation import EvaluationAgent
from automl_agent.agents.feature import FeatureAgent
from automl_agent.types import CandidateResult, DataBundle, FeaturePlan


class ModelSearchAgent(BaseAgent):
    name = "Model Search Agent"

    def __init__(self, max_workers: int = 4, cv_splits: int = 3, random_state: int = 42) -> None:
        super().__init__()
        self.max_workers = max_workers
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.evaluator = EvaluationAgent()

    def search(self, data: DataBundle, features: FeaturePlan) -> list[CandidateResult]:
        candidates = self._candidates(data.task_type)
        self.log(f"Training {len(candidates)} candidate models with up to {self.max_workers} workers.")
        results: list[CandidateResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
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
            cv_score = self._cv_score(pipeline, data)
            pipeline.fit(data.X_train, data.y_train)
            result = self.evaluator.evaluate(name, pipeline, data)
            result.cv_score = cv_score
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

    def _cv_score(self, pipeline: Pipeline, data: DataBundle) -> Optional[float]:
        cv = self._cv_splitter(data)
        if cv is None:
            return None
        scores = cross_val_score(
            pipeline,
            data.X_train,
            data.y_train,
            cv=cv,
            scoring=self.evaluator.scoring(data.task_type),
            n_jobs=1,
        )
        return float(scores.mean())

    def _cv_splitter(self, data: DataBundle):
        if data.task_type == "classification":
            n_splits = min(self.cv_splits, int(data.y_train.value_counts().min()))
            if n_splits < 2:
                return None
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        if len(data.y_train) < 2 * self.cv_splits:
            return None
        return KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

    def _candidates(self, task_type: str) -> Dict[str, object]:
        if task_type == "classification":
            return {
                "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "extra_trees": ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=-1),
                "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
                "svc_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced"),
            }
        return {
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "extra_trees": ExtraTreesRegressor(n_estimators=120, random_state=42, n_jobs=-1),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
            "svr_rbf": SVR(kernel="rbf"),
        }
