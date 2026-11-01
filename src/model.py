"""The MediaPipe pose model file: where it lives and how to fetch it.

The ``.task`` file (about 9 MB) is not stored in git. ``ensure_model()`` downloads
it on first use, so a fresh checkout or a new Hugging Face Space starts without a
manual step. ``scripts/download_model.py`` is the command-line wrapper.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"
MIN_PLAUSIBLE_BYTES = 1_000_000     # a real model is ~9 MB; anything tiny is an error page or a cut-off download
DOWNLOAD_TIMEOUT_S = 60


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL, force: bool = False) -> Path:
    """Return the model path, downloading the file first if it is missing (or ``force``).

    Raises ``OSError`` if the download fails or the file looks wrong; the half-written
    file is removed so the next call retries cleanly.
    """
    if path.exists() and not force and path.stat().st_size >= MIN_PLAUSIBLE_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if partial.stat().st_size < MIN_PLAUSIBLE_BYTES:
            raise OSError(f"downloaded file is only {partial.stat().st_size} bytes; expected the model")
        partial.replace(path)
    except OSError:
        partial.unlink(missing_ok=True)
        raise
    return path
