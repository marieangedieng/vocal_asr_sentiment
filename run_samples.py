from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.pipeline import VoiceSentimentPipeline


def main() -> None:
    pipeline = VoiceSentimentPipeline()
    for audio_path in sorted(Path("data/samples").glob("*.mp3")):
        result = pipeline.predict_path(audio_path)
        print(audio_path.name)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

