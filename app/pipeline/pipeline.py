from __future__ import annotations

from pathlib import Path

from app.pipeline.asr import Wav2VecFrenchASR
from app.pipeline.config import PipelineConfig
from app.pipeline.preprocessing import AudioData, load_and_preprocess_audio
from app.pipeline.sentiment import FrenchSentimentClassifier


class VoiceSentimentPipeline:
    """Complete audio preprocessing, ASR and sentiment inference pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.asr = Wav2VecFrenchASR(self.config)
        self.sentiment = FrenchSentimentClassifier(self.config)

    def predict_path(self, audio_path: str | Path) -> dict:
        audio: AudioData = load_and_preprocess_audio(audio_path, self.config)
        transcription = self.asr.transcribe(audio.waveform, audio.sample_rate)
        sentiment = self.sentiment.predict(transcription)
        return {
            "transcription": transcription,
            "sentiment": sentiment.label,
            "confidence": sentiment.confidence,
            "duration_sec": round(audio.duration_sec, 3),
        }

