from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split

from automl_agent.agents.base import BaseAgent
from automl_agent.types import DataBundle, DatasetProfile, TaskType


BUILT_IN_DATASETS = {
    "iris": (load_iris, "target", "classification"),
    "breast_cancer": (load_breast_cancer, "target", "classification"),
    "wine": (load_wine, "target", "classification"),
    "diabetes": (load_diabetes, "target", "regression"),
}


class DataAgent(BaseAgent):
    name = "Data Agent"

    def load(
        self,
        dataset: Optional[str] = None,
        csv_path: Optional[Path] = None,
        target: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> DataBundle:
        if csv_path:
            df, dataset_name, resolved_target = self._load_csv(csv_path, target)
        else:
            df, dataset_name, resolved_target, built_in_task = self._load_builtin(dataset or "breast_cancer")
            task_type = task_type or built_in_task

        if resolved_target not in df.columns:
            raise ValueError(f"Target column '{resolved_target}' was not found.")

        df = self._clean(df, resolved_target)
        inferred_task = task_type or self._infer_task(df[resolved_target])
        profile = self._profile(df, resolved_target, inferred_task)
        X = df.drop(columns=[resolved_target])
        y = df[resolved_target]
        stratify = y if inferred_task == "classification" and y.nunique() > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        self.log(f"Loaded {dataset_name} with {len(df)} rows and target '{resolved_target}'.")
        return DataBundle(
            dataset_name=dataset_name,
            target=resolved_target,
            task_type=inferred_task,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            profile=profile,
        )

    def _load_csv(self, csv_path: Path, target: Optional[str]) -> tuple[pd.DataFrame, str, str]:
        df = pd.read_csv(csv_path)
        resolved_target = target or df.columns[-1]
        return df, csv_path.stem, resolved_target

    def _load_builtin(self, dataset: str) -> tuple[pd.DataFrame, str, str, TaskType]:
        if dataset not in BUILT_IN_DATASETS:
            valid = ", ".join(sorted(BUILT_IN_DATASETS))
            raise ValueError(f"Unknown built-in dataset '{dataset}'. Choose one of: {valid}.")
        loader, target, task_type = BUILT_IN_DATASETS[dataset]
        raw = loader(as_frame=True)
        df = raw.frame.copy()
        if target not in df.columns:
            df[target] = raw.target
        return df, dataset, target, task_type  # type: ignore[return-value]

    def _clean(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        cleaned = df.dropna(subset=[target])
        if len(cleaned) < len(df):
            self.log(f"Dropped {len(df) - len(cleaned)} rows with a missing target value.")
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        if len(cleaned) < before:
            self.log(f"Dropped {before - len(cleaned)} duplicate rows.")

        feature_columns = [column for column in cleaned.columns if column != target]
        constant = [column for column in feature_columns if cleaned[column].nunique(dropna=False) <= 1]
        id_like = [
            column
            for column in feature_columns
            if column not in constant
            and not pd.api.types.is_numeric_dtype(cleaned[column])
            and cleaned[column].nunique(dropna=True) == len(cleaned)
        ]
        uninformative = constant + id_like
        if uninformative:
            cleaned = cleaned.drop(columns=uninformative)
            self.log(f"Dropped uninformative columns: {', '.join(uninformative)}.")
        return cleaned

    def _infer_task(self, target: pd.Series) -> TaskType:
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"
        unique = target.nunique(dropna=True)
        ratio = unique / max(len(target), 1)
        return "classification" if unique <= 20 and ratio < 0.2 else "regression"

    def _profile(self, df: pd.DataFrame, target: str, task_type: TaskType) -> DatasetProfile:
        feature_df = df.drop(columns=[target])
        numeric_features = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [column for column in feature_df.columns if column not in numeric_features]
        y = df[target]
        if task_type == "classification":
            target_summary = {"classes": int(y.nunique(dropna=True)), "counts": y.value_counts().head(20).to_dict()}
        else:
            target_summary = {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            }
        return DatasetProfile(
            rows=len(df),
            columns=len(df.columns),
            target=target,
            task_type=task_type,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            missing_values=df.isna().sum().to_dict(),
            target_summary=target_summary,
        )

