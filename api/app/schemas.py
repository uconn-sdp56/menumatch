from typing import List
from pydantic import BaseModel, confloat


class ClassifiedItem(BaseModel):
    id: int
    name: str
    confidence: confloat(ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    items: List[ClassifiedItem]
    explanation: str


class PortionEstimate(BaseModel):
    id: int
    name: str
    num_servings: confloat(ge=0.0, le=10.0)


class PortionEstimationResult(BaseModel):
    servings: List[PortionEstimate]
    explanation: str