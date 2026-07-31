from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.config import PipelineConfig
from app.pipeline.errors import SilentAudioError, UnsupportedAudioFormatError
from app.pipeline.preprocessing import load_and_preprocess_audio
from app.pipeline.sentiment import FrenchSentimentClassifier


def test_preprocessing_resamples_normalizes_and_reports_duration(tmp_path: Path) -> None:
    sample_rate = 8000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, waveform, sample_rate)

    audio = load_and_preprocess_audio(audio_path, PipelineConfig(silence_rms_threshold=0.001))

    assert audio.sample_rate == 16000
    assert audio.duration_sec == pytest.approx(duration, abs=0.02)
    assert np.max(np.abs(audio.waveform)) == pytest.approx(1.0, abs=0.02)


def test_preprocessing_rejects_unsupported_extension(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.txt"
    audio_path.write_text("not audio")

    with pytest.raises(UnsupportedAudioFormatError):
        load_and_preprocess_audio(audio_path)


def test_preprocessing_rejects_silence(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)

    with pytest.raises(SilentAudioError):
        load_and_preprocess_audio(audio_path)


def test_sentiment_mapping_aggregates_star_probabilities() -> None:
    mapped = FrenchSentimentClassifier._probabilities_by_star(
        [0.05, 0.10, 0.20, 0.25, 0.40],
        {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"},
    )
    grouped_positive = mapped[4] + mapped[5]
    grouped_negative = mapped[1] + mapped[2]

    assert grouped_positive == pytest.approx(0.65)
    assert grouped_negative == pytest.approx(0.15)

