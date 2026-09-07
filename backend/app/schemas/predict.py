from pydantic import BaseModel, Field
from typing import Dict, List


class PredictResponse(BaseModel):
    predicted_class: str = Field(..., examples=["Good"])
    confidence: float = Field(..., ge=0, le=1, examples=[0.91])
    probabilities: Dict[str, float]


class HistoryItem(BaseModel):
    id: str
    filename: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    created_at: str


class HistoryResponse(BaseModel):
    total: int
    items: List[HistoryItem]