from pathlib import Path

import pandas as pd

from automl_agent.agents.data import DataAgent


def _write_csv(tmp_path: Path) -> Path:
    rows = []
    for index in range(40):
        rows.append(
            {
                "record_id": f"user-{index}",
                "constant": 1,
                "feature_a": index % 7,
                "feature_b": float(index) / 3.0,
                "label": index % 2,
            }
        )
    # Duplicate row and a row with a missing target.
    rows.append(dict(rows[0]))
    rows.append({"record_id": "user-x", "constant": 1, "feature_a": 1, "feature_b": 0.5, "label": None})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_data_agent_cleans_csv(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    agent = DataAgent()
    data = agent.load(csv_path=csv_path, target="label")

    columns = set(data.X_train.columns)
    assert "constant" not in columns
    assert "record_id" not in columns
    assert {"feature_a", "feature_b"} <= columns
    # 42 raw rows minus one duplicate and one missing-target row.
    assert len(data.X_train) + len(data.X_test) == 40
    assert data.task_type == "classification"


def test_data_agent_infers_regression() -> None:
    agent = DataAgent()
    data = agent.load(dataset="diabetes")
    assert data.task_type == "regression"
