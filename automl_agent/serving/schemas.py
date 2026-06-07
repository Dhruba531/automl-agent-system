from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    model_name: str
    task_type: str
    predictions: List[Any]
    probabilities: Optional[List[Any]] = None

