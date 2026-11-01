"""Turn one of your own clips into a small example clip for the app (and check it works).

Usage:
    python scripts/prepare_example.py my_clip.mov forward_defence --name nets_1 [--start 1.5] [--seconds 5]

Writes ``examples/<shot_type>_<name>.mp4``: trimmed, scaled to 720p, no audio, H.264, so the
file is a few MB. Then runs the full analysis on it and prints what a visitor would see, so you
know the example will not end with a "please film again" message.

Only use footage of people who agreed to have it shown publicly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.overlay import _ffmpeg_exe
from src.pipeline import SHOT_TYPES, analyze

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def main() -> None:
    """Parse the command line, write the example clip and report how the analysis sees it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="your original clip")
    parser.add_argument("shot_type", choices=SHOT_TYPES)
    parser.add_argument("--name", default="demo", help="short name used in the file name (default: demo)")
    parser.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start")
    parser.add_argument("--seconds", type=float, default=5.0, help="length to keep (default: 5)")
    parser.add_argument("--out-dir", type=Path, default=EXAMPLES_DIR, help="where to write (default: examples/)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.shot_type}_{args.name}.mp4"
    command = [_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(args.start), "-t", str(args.seconds),
               "-i", str(args.video), "-an", "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        sys.exit(f"ffmpeg failed: {done.stderr.strip()}")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    result = analyze(out, args.shot_type, render=False)
    if result.ok:
        print(f"Analysis OK: score {result.score.total:.0f}/100 (confidence {result.score.confidence:.2f}). "
              "This is a good example clip.")
    else:
        print(f"The app would say: {result.message}")
        print("Pick a different part of the clip (--start/--seconds) or film it again before using it as an example.")


if __name__ == "__main__":
    main()
