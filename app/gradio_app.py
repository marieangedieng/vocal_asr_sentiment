from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def predict_from_api(audio_path: str | None) -> tuple[str, str]:
    """Send the audio file to FastAPI and format the response for Gradio."""
    if not audio_path:
        return "", "Aucun fichier audio fourni."

    path = Path(audio_path)
    try:
        with path.open("rb") as audio_file:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (path.name, audio_file, "application/octet-stream")},
                timeout=600,
            )
    except requests.RequestException as exc:
        return "", f"API indisponible ({API_URL}) : {exc}"

    if response.status_code != 200:
        try:
            detail: Any = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return "", f"Erreur API {response.status_code}: {detail}"

    payload = response.json()
    transcription = payload["transcription"]
    sentiment = payload["sentiment"]
    confidence = payload["confidence"]
    colors = {"positif": "#157347", "neutre": "#6c757d", "negatif": "#bb2d3b"}
    html = (
        f"<div style='font-size:1.2rem;color:{colors.get(sentiment, '#333')};'>"
        f"<strong>{sentiment.upper()}</strong> - confiance {confidence:.2%}"
        f"</div><div>Duree analysee: {payload['duration_sec']:.2f}s</div>"
    )
    return transcription, html


def build_interface():
    import gradio as gr

    with gr.Blocks(title="Sentiment vocal client") as demo:
        gr.Markdown("# Detection automatique de sentiment vocal")
        audio = gr.File(
            label="Fichier audio client",
            file_types=[".wav", ".mp3"],
            type="filepath",
        )
        submit = gr.Button("Analyser")
        transcription = gr.Textbox(label="Transcription ASR", lines=4)
        sentiment = gr.HTML(label="Sentiment")
        submit.click(fn=predict_from_api, inputs=audio, outputs=[transcription, sentiment])
    return demo


if __name__ == "__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860, share=False)

