from dataclasses import dataclass
import os
from pathlib import Path

import torch


ASR_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
SENTIMENT_MODEL_ID = "cmarkea/distilcamembert-base-sentiment"

LOCAL_CACHE_DIR = Path(os.getenv("HF_HOME", ".cache/huggingface")).resolve()
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(LOCAL_CACHE_DIR / "transformers"))


@dataclass(frozen=True)
class PipelineConfig:
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    max_duration_sec: float = float(os.getenv("MAX_DURATION_SEC", "300"))
    silence_rms_threshold: float = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.003"))
    asr_model_id: str = os.getenv("ASR_MODEL_ID", ASR_MODEL_ID)
    sentiment_model_id: str = os.getenv("SENTIMENT_MODEL_ID", SENTIMENT_MODEL_ID)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.float32
