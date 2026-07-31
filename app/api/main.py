from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorResponse, HealthResponse, PredictionResponse
from app.pipeline.config import PipelineConfig
from app.pipeline.errors import PipelineError
from app.pipeline.pipeline import VoiceSentimentPipeline
from app.pipeline.preprocessing import save_upload_to_temp


config = PipelineConfig()
_pipeline: VoiceSentimentPipeline | None = None

app = FastAPI(
    title="Detection automatique de sentiment vocal",
    description="Pipeline francais: audio -> Wav2Vec2 ASR -> DistilCamemBERT sentiment.",
    version="1.0.0",
)


def get_pipeline() -> VoiceSentimentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VoiceSentimentPipeline(config)
    return _pipeline


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_, exc: PipelineError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        asr_model=config.asr_model_id,
        sentiment_model=config.sentiment_model_id,
        models_loaded=_pipeline is not None,
        device=str(config.device),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    temp_path: Path | None = None
    try:
        temp_path = await save_upload_to_temp(file)
        result = get_pipeline().predict_path(temp_path)
        return PredictionResponse(**result)
    except PipelineError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur interne pendant la prediction: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

