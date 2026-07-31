from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
from fastapi import UploadFile
from scipy.signal import resample_poly

from app.pipeline.config import PipelineConfig
from app.pipeline.errors import (
    AudioTooLongError,
    CorruptedAudioError,
    EmptyAudioError,
    SilentAudioError,
    UnsupportedAudioFormatError,
)


SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


@dataclass(frozen=True)
class AudioData:
    waveform: np.ndarray
    sample_rate: int
    duration_sec: float
    rms: float


def validate_audio_filename(filename: str) -> str:
    """Return the normalized extension or raise a user-facing error."""
    ext = Path(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedAudioFormatError(f"Format audio non supporte. Formats acceptes: {allowed}.")
    return ext


async def save_upload_to_temp(upload: UploadFile) -> Path:
    """Persist an uploaded audio file to a temporary path for librosa decoding."""
    validate_audio_filename(upload.filename or "")
    content = await upload.read()
    if not content:
        raise EmptyAudioError("Le fichier audio est vide.")

    suffix = Path(upload.filename or "").suffix.lower()
    temp = NamedTemporaryFile(delete=False, suffix=suffix)
    with temp:
        temp.write(content)
    return Path(temp.name)


def load_and_preprocess_audio(path: str | Path, config: PipelineConfig | None = None) -> AudioData:
    """Load, convert to mono 16 kHz, normalize amplitude and reject invalid audio."""
    cfg = config or PipelineConfig()
    audio_path = Path(path)
    validate_audio_filename(audio_path.name)

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise EmptyAudioError("Le fichier audio est vide.")

    try:
        waveform, sample_rate = _decode_audio(audio_path, cfg.sample_rate)
    except Exception as exc:
        raise CorruptedAudioError("Le fichier audio est corrompu ou illisible.") from exc

    if waveform.size == 0:
        raise EmptyAudioError("Le fichier audio ne contient aucun echantillon exploitable.")

    waveform = waveform.astype(np.float32, copy=False)
    duration_sec = float(waveform.shape[0] / sample_rate)
    if duration_sec > cfg.max_duration_sec:
        raise AudioTooLongError(
            f"Le fichier dure {duration_sec:.1f}s, au-dela de la limite de {cfg.max_duration_sec:.0f}s."
        )

    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < cfg.silence_rms_threshold:
        raise SilentAudioError("Audio silencieux detecte: energie RMS trop faible pour transcrire.")

    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform / peak

    return AudioData(waveform=waveform, sample_rate=sample_rate, duration_sec=duration_sec, rms=rms)


def _decode_audio(audio_path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    """Decode audio with a fast WAV path and a broader MP3 fallback."""
    try:
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        waveform, sample_rate = librosa.load(audio_path, sr=target_sample_rate, mono=True)
        return waveform.astype(np.float32, copy=False), target_sample_rate

    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)

    if sample_rate != target_sample_rate:
        gcd = np.gcd(sample_rate, target_sample_rate)
        waveform = resample_poly(waveform, target_sample_rate // gcd, sample_rate // gcd)
        sample_rate = target_sample_rate

    return waveform.astype(np.float32, copy=False), sample_rate
