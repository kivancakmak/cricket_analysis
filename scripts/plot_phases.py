"""Debug plot: velocity curves with the detected key moments marked.

Usage:
    python scripts/plot_phases.py data/raw/clip.mp4 [more.mp4 ...] [--out-dir outputs/plots]

Pose extraction is cached in ``outputs/cache/`` so re-running while tuning
``PHASE_CONFIG`` is fast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orientation import detect_orientation
from src.phases import PHASE_CONFIG, detect_key_moments
from src.pose import PoseSequence, extract_pose, smooth_pose

COLORS = {"STANCE": "tab:green", "FRONT FOOT DOWN": "tab:blue", "CONTACT": "tab:red"}


def load_pose(video: Path, cache_dir: Path) -> PoseSequence:
    cache = cache_dir / f"{video.stem}.npz"
    if cache.exists():
        return PoseSequence.load(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq = extract_pose(video)
    seq.save(cache)
    return seq


def plot(video: Path, out_dir: Path, cache_dir: Path) -> Path:
    seq = smooth_pose(load_pose(video, cache_dir))
    orientation = detect_orientation(seq)
    moments = detect_key_moments(seq, orientation)
    sig = moments.signals
    if sig is None:
        raise SystemExit(f"{video.name}: {moments.notes}")
    cfg = PHASE_CONFIG

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = sig.time_s
    axes[0].plot(t, sig.ankle_speed, label="front ankle speed", color="tab:purple")
    axes[0].plot(t, sig.hip_speed, label="hip speed", color="tab:orange")
    axes[0].axhline(cfg["stance_ankle_speed"], color="tab:purple", ls=":", lw=1)
    axes[0].axhline(cfg["stance_hip_speed"], color="tab:orange", ls=":", lw=1)
    axes[0].set_title("Stance: ankle and hip speed (dotted = 'still' thresholds)")
    axes[1].plot(t, sig.ankle_forward, label="front ankle forward", color="tab:blue")
    axes[1].plot(t, sig.ankle_vertical, label="front ankle vertical (down +)", color="tab:gray")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Landing: front ankle forward / vertical velocity")
    axes[2].plot(t, sig.wrist_forward, label="wrists forward", color="tab:red")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("Contact: forward hand velocity")

    top = axes[0].get_ylim()[1]
    found = {"STANCE": moments.stance, "FRONT FOOT DOWN": moments.front_foot_landing,
             "CONTACT": moments.contact}
    for row, (label, moment) in enumerate((k, v) for k, v in found.items() if v is not None):
        for ax in axes:
            ax.axvline(moment.time_s, color=COLORS[label], lw=1.5, ls="--")
        # stagger the captions vertically so neighbouring moments do not overlap
        axes[0].annotate(f"{label} f{moment.frame} (conf {moment.confidence:.2f})",
                         (moment.time_s, top * (0.97 - 0.1 * row)), xytext=(4, 0),
                         textcoords="offset points", color=COLORS[label], fontsize=8,
                         va="top", ha="left")
    for ax in axes:
        ax.set_ylabel("body heights / s")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{video.stem[:60]}\nfacing {orientation.facing}, front leg {orientation.front_leg}")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{video.stem}_phases.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)

    print(f"{video.name}")
    for label, frame in moments.as_key_frames().items():
        print(f"  {label:16s} frame {frame:4d}  t={frame / seq.fps:5.2f}s")
    for note in moments.notes:
        print(f"  note: {note}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache"))
    args = parser.parse_args()
    for video in args.videos:
        print(f"  -> {plot(video, args.out_dir, args.cache_dir)}")


if __name__ == "__main__":
    main()
