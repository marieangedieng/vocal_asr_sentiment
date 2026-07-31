from __future__ import annotations

import torch
from app.pipeline.config import PipelineConfig


class Wav2VecFrenchASR:
    """Thin wrapper around the French Wav2Vec2 CTC model."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        import torch
        from transformers import AutoModelForCTC, Wav2Vec2Processor

        self.config = config or PipelineConfig()
        self.processor = Wav2Vec2Processor.from_pretrained(self.config.asr_model_id)
        self.model = AutoModelForCTC.from_pretrained(
            self.config.asr_model_id,
            torch_dtype=self.config.torch_dtype,
        )
        self.model.to(self.config.device)
        self.model.eval()

    def transcribe(self, waveform, sample_rate: int) -> str:
        import torch

        with torch.inference_mode():
            return self._transcribe(waveform, sample_rate)

    def _transcribe(self, waveform, sample_rate: int) -> str:
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.config.device)
        if self.config.device.type == "cuda":
            input_values = input_values.half()

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return " ".join(transcription.strip().lower().split())
