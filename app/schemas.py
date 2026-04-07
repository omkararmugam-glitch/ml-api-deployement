from pydantic import BaseModel, Field
from typing import List


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10)
    sepal_width: float = Field(..., gt=0, le=10)
    petal_length: float = Field(..., gt=0, le=10)
    petal_width: float = Field(..., gt=0, le=10)


class PredictionResponse(BaseModel):
    prediction: int
    species: str
    confidence: float
    probabilities: dict


class BatchRequest(BaseModel):
    instances: List[IrisFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    name: str
    version: str
    features: List[str]
    classes: List[str]