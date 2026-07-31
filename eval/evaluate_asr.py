from __future__ import annotations

import argparse
import os
from itertools import islice
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache/numba").resolve()))

import numpy as np
from datasets import load_dataset
from jiwer import wer
from scipy.signal import resample_poly

from app.pipeline.asr import Wav2VecFrenchASR
from app.pipeline.config import PipelineConfig
from app.pipeline.preprocessing import AudioData


RESULTS_PATH = Path("eval/results.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR on a small Common Voice sample.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of audio samples to evaluate.")
    parser.add_argument("--dataset", default="fsicoli/common_voice_17_0", help="Hugging Face dataset id.")
    parser.add_argument("--config", default="fr", help="Dataset language/configuration.")
    parser.add_argument("--split", default="test", help="Dataset split to read.")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download the selected split instead of streaming it.",
    )
    return parser.parse_args()


def main(
    limit: int = 25,
    dataset_id: str = "fsicoli/common_voice_17_0",
    dataset_config: str = "fr",
    split: str = "test",
    streaming: bool = True,
) -> None:
    config = PipelineConfig()
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        split=split if streaming else f"{split}[:{limit}]",
        streaming=streaming,
        trust_remote_code=True,
    )
    asr = Wav2VecFrenchASR(config)

    references: list[str] = []
    predictions: list[str] = []
    samples = islice(dataset, limit) if streaming else dataset
    for sample in samples:
        audio = sample["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        if sample_rate != config.sample_rate:
            gcd = int(np.gcd(sample_rate, config.sample_rate))
            waveform = resample_poly(waveform, config.sample_rate // gcd, sample_rate // gcd)
            sample_rate = config.sample_rate

        data = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            duration_sec=len(waveform) / sample_rate,
            rms=0.0,
        )
        references.append(sample["sentence"].lower())
        predictions.append(asr.transcribe(data.waveform, data.sample_rate))

    score = wer(references, predictions)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write("\n## Evaluation ASR - Common Voice FR\n\n")
        file.write(f"- Echantillons: {len(references)}\n")
        file.write(f"- WER: {score:.4f}\n")

    print(f"WER={score:.4f} sur {len(references)} echantillons")


if __name__ == "__main__":
    args = parse_args()
    main(
        limit=args.limit,
        dataset_id=args.dataset,
        dataset_config=args.config,
        split=args.split,
        streaming=not args.no_streaming,
    )
