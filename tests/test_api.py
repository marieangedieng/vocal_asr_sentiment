from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.api import main


class FakePipeline:
    def predict_path(self, audio_path: Path) -> dict:
        return {
            "transcription": "je suis satisfait du service",
            "sentiment": "positif",
            "confidence": 0.91,
            "duration_sec": 1.0,
        }


def test_health_endpoint() -> None:
    client = TestClient(main.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_valid_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())
    audio_path = tmp_path / "valid.wav"
    sf.write(audio_path, np.sin(np.linspace(0, 100, 16000)).astype(np.float32), 16000)

    client = TestClient(main.app)
    with audio_path.open("rb") as audio_file:
        response = client.post("/predict", files={"file": ("valid.wav", audio_file, "audio/wav")})

    assert response.status_code == 200
    assert response.json()["sentiment"] == "positif"


def test_predict_endpoint_invalid_extension() -> None:
    client = TestClient(main.app)
    response = client.post("/predict", files={"file": ("invalid.txt", b"hello", "text/plain")})

    assert response.status_code == 415
    assert "Format audio non supporte" in response.json()["detail"]

