from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from app.pipeline.config import PipelineConfig


SENTIMENT_BY_STAR = {
    1: "negatif",
    2: "negatif",
    3: "neutre",
    4: "positif",
    5: "positif",
}


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    star_probabilities: dict[int, float]


class FrenchSentimentClassifier:
    """Map the model's 1-5 star probabilities to three business sentiment classes."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config or PipelineConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.sentiment_model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.sentiment_model_id,
            torch_dtype=self.config.torch_dtype,
        )
        self.model.to(self.config.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> SentimentResult:
        clean_text = text.strip()
        if not clean_text:
            return SentimentResult(label="neutre", confidence=0.0, star_probabilities={})

        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0).float().cpu().tolist()

        id2label = getattr(self.model.config, "id2label", {})
        star_probabilities = self._probabilities_by_star(probabilities, id2label)
        grouped = {"negatif": 0.0, "neutre": 0.0, "positif": 0.0}
        for star, probability in star_probabilities.items():
            grouped[SENTIMENT_BY_STAR[star]] += probability

        label = max(grouped, key=grouped.get)
        return SentimentResult(
            label=label,
            confidence=round(float(grouped[label]), 4),
            star_probabilities=star_probabilities,
        )

    @staticmethod
    def _probabilities_by_star(probabilities: list[float], id2label: dict[int, str]) -> dict[int, float]:
        mapped: dict[int, float] = {}
        for index, probability in enumerate(probabilities):
            raw_label = str(id2label.get(index, index + 1))
            match = re.search(r"([1-5])", raw_label)
            star = int(match.group(1)) if match else index + 1
            if star in SENTIMENT_BY_STAR:
                mapped[star] = float(probability)
        return mapped
