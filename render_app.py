from __future__ import annotations

import os
import threading
import time

import uvicorn

from app.gradio_app import build_interface


def run_api() -> None:
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, log_level="info")


os.environ.setdefault("API_URL", "http://127.0.0.1:8000")
threading.Thread(target=run_api, daemon=True).start()
time.sleep(3)

port = int(os.getenv("PORT", "7860"))
build_interface().launch(server_name="0.0.0.0", server_port=port, share=False)
