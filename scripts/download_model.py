"""Download the MediaPipe pose_landmarker_full.task model into ``models/``.

Usage:
    python scripts/download_model.py [--force]

The app also does this automatically on startup when the file is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import MODEL_PATH, MODEL_URL, ensure_model


def main() -> None:
    """Parse the command line and download the model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if MODEL_PATH.exists() and not args.force:
        print(f"Model already present: {MODEL_PATH}")
        return
    print(f"Downloading {MODEL_URL}")
    try:
        path = ensure_model(force=args.force)
    except OSError as exc:  # URLError / HTTPError / timeouts are OSError subclasses
        sys.exit(f"Download failed: {exc}")
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
