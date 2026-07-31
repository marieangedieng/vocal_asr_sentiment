from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    asr_model: str
    sentiment_model: str
    models_loaded: bool
    device: str


class PredictionResponse(BaseModel):
    transcription: str
    sentiment: Literal["positif", "neutre", "negatif"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    duration_sec: float = Field(..., ge=0.0)


class ErrorResponse(BaseModel):
    detail: str

