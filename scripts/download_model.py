"""Download the MediaPipe Pose Landmarker (full) model into ``models/``.

Usage:
    python scripts/download_model.py [--force]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"


def download(force: bool = False) -> Path:
    """Fetch the model unless it already exists; return its path."""
    if MODEL_PATH.exists() and not force:
        print(f"Model already present: {MODEL_PATH}")
        return MODEL_PATH

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".task.part")
    print(f"Downloading {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
    except OSError as exc:  # URLError / HTTPError are OSError subclasses
        tmp_path.unlink(missing_ok=True)
        sys.exit(f"Download failed: {exc}")
    tmp_path.replace(MODEL_PATH)
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    return MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    download(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
