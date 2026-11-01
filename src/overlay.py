"""Stage 3: skeleton overlay video and key-frame images.

Draws the pose skeleton on every frame, labels key moments and holds each one
for about half a second. Frames are piped straight into ffmpeg (libx264,
yuv420p, faststart) so the result plays in any browser.

CLI:
    python -m src.overlay data/sample.mp4 [--out outputs/x.mp4] [--key STANCE=12 ...]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from src.pose import PoseSequence, extract_pose, smooth_pose

# Body connections (MediaPipe landmark indices). Face landmarks are omitted: a
# clean torso/limb skeleton is easier to read, and a face mesh adds nothing here.
CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (23, 25), (25, 27), (24, 26), (26, 28),   # legs
    (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32),  # feet
)
JOINTS: tuple[int, ...] = tuple(range(11, 33))

# BGR colours: left side, right side, centre line.
COLOR_LEFT = (80, 200, 255)
COLOR_RIGHT = (255, 160, 60)
COLOR_CENTER = (230, 230, 230)
LABEL_BG = (30, 30, 30)
LABEL_FG = (255, 255, 255)

VISIBILITY_THRESHOLD = 0.5   # below this a joint/limb is drawn faded
FADE_ALPHA = 0.3
HOLD_SECONDS = 0.5
MAX_WIDTH = 1280             # output is downscaled to this width if larger


def _side_color(a: int, b: int) -> tuple[int, int, int]:
    if a % 2 == 1 and b % 2 == 1:
        return COLOR_LEFT
    if a % 2 == 0 and b % 2 == 0:
        return COLOR_RIGHT
    return COLOR_CENTER


def draw_skeleton(frame: np.ndarray, landmarks: np.ndarray | None,
                  vis_threshold: float = VISIBILITY_THRESHOLD) -> np.ndarray:
    """Draw the skeleton on ``frame`` in place and return it.

    ``landmarks`` is a (33, 4) array of normalized x, y, z, visibility; None
    (no person detected) leaves the frame untouched. Joints and limbs with
    visibility below ``vis_threshold`` are drawn faded.
    """
    if landmarks is None:
        return frame
    h, w = frame.shape[:2]
    thickness = max(2, h // 240)
    radius = max(3, h // 150)
    pts = np.where(np.isnan(landmarks[:, :2]), -1, landmarks[:, :2] * (w, h))
    valid = ~np.isnan(landmarks[:, 0])
    vis = np.nan_to_num(landmarks[:, 3])

    def draw(target: np.ndarray, faded: bool) -> None:
        """Draw the limbs and joints that are (not) faded onto ``target``."""
        for a, b in CONNECTIONS:
            if not (valid[a] and valid[b]):
                continue
            if (min(vis[a], vis[b]) < vis_threshold) != faded:
                continue
            cv2.line(target, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     _side_color(a, b), thickness, cv2.LINE_AA)
        for j in JOINTS:
            if not valid[j] or (vis[j] < vis_threshold) != faded:
                continue
            cv2.circle(target, tuple(pts[j].astype(int)), radius, _side_color(j, j),
                       -1, cv2.LINE_AA)
            cv2.circle(target, tuple(pts[j].astype(int)), radius, LABEL_BG, 1, cv2.LINE_AA)

    if (vis[valid] < vis_threshold).any():
        layer = frame.copy()
        draw(layer, faded=True)
        cv2.addWeighted(layer, FADE_ALPHA, frame, 1 - FADE_ALPHA, 0, dst=frame)
    draw(frame, faded=False)
    return frame


def _draw_banner(frame: np.ndarray, text: str) -> None:
    """Centered label at the top of the frame."""
    h, w = frame.shape[:2]
    scale = h / 600
    thick = max(2, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    pad = round(12 * scale)
    x0, y0 = (w - tw) // 2 - pad, round(16 * scale)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 2 * pad, y0 + th + base + 2 * pad), LABEL_BG, -1)
    cv2.putText(frame, text, (x0 + pad, y0 + pad + th), cv2.FONT_HERSHEY_DUPLEX,
                scale, LABEL_FG, thick, cv2.LINE_AA)


def _draw_notes(frame: np.ndarray, lines: Sequence[str]) -> None:
    """Annotation lines in the bottom-left corner."""
    h = frame.shape[0]
    scale = h / 900
    thick = max(1, round(scale * 2))
    step = round(34 * scale)
    y = h - round(20 * scale) - step * (len(lines) - 1)
    for line in lines:
        cv2.putText(frame, line, (round(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1,
                    LABEL_BG, thick + 3, cv2.LINE_AA)
        cv2.putText(frame, line, (round(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1,
                    LABEL_FG, thick, cv2.LINE_AA)
        y += step


def _prepare(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale if wider than ``max_width`` and force even dimensions (yuv420p)."""
    h, w = frame.shape[:2]
    if w > max_width:
        h = round(h * max_width / w)
        w = max_width
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame[: h - h % 2, : w - w % 2]


def _compose(frame: np.ndarray, seq: PoseSequence, index: int, labels: Sequence[str],
             notes: Sequence[str], max_width: int) -> np.ndarray:
    frame = _prepare(frame, max_width)
    landmarks = seq.frame(index) if index < seq.n_frames else None
    draw_skeleton(frame, landmarks)
    if labels:
        _draw_banner(frame, " / ".join(labels))
    if notes:
        _draw_notes(frame, notes)
    return frame


def _ffmpeg_exe() -> str:
    """System ffmpeg if installed, otherwise the binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _labels_by_frame(key_frames: Mapping[str, int] | None) -> dict[int, list[str]]:
    by_frame: dict[int, list[str]] = {}
    for label, idx in (key_frames or {}).items():
        by_frame.setdefault(int(idx), []).append(label)
    return by_frame


def render_overlay(
    video_path: str | Path,
    pose_seq: PoseSequence,
    out_path: str | Path,
    key_frames: Mapping[str, int] | None = None,
    annotations: Mapping[int, Sequence[str]] | None = None,
    max_width: int = MAX_WIDTH,
) -> Path:
    """Write a browser-playable H.264 mp4 with the skeleton drawn on every frame.

    Args:
        key_frames: label -> frame index, e.g. ``{"STANCE": 12, "CONTACT": 48}``.
            Each is captioned and held for about half a second.
        annotations: frame index -> text lines shown bottom-left on that frame
            (and during its hold, if it is a key frame).
        max_width: wider videos are downscaled to this width.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    labels = _labels_by_frame(key_frames)
    notes = annotations or {}
    hold = max(1, round(HOLD_SECONDS * pose_seq.fps))
    proc: subprocess.Popen | None = None
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _compose(frame, pose_seq, i, labels.get(i, ()), notes.get(i, ()), max_width)
            if proc is None:
                h, w = frame.shape[:2]
                proc = subprocess.Popen(
                    [_ffmpeg_exe(), "-y", "-loglevel", "error",
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                     "-r", f"{pose_seq.fps}", "-i", "-",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            data = np.ascontiguousarray(frame).tobytes()
            for _ in range(hold if i in labels else 1):
                proc.stdin.write(data)
            i += 1
    finally:
        cap.release()
        if proc is not None:
            proc.stdin.close()
            stderr = proc.stderr.read().decode(errors="replace")
            proc.stderr.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr.strip()}")
    if proc is None:
        raise ValueError(f"No frames could be read from {video_path}")
    return out_path


def export_key_frames(
    video_path: str | Path,
    pose_seq: PoseSequence,
    key_frames: Mapping[str, int],
    out_dir: str | Path,
    max_width: int = MAX_WIDTH,
) -> dict[str, Path]:
    """Save each key frame (skeleton + label) as a PNG; returns label -> path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = _labels_by_frame(key_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    paths: dict[str, Path] = {}
    try:
        for i in range(max(wanted, default=-1) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if i not in wanted:
                continue
            image = _compose(frame, pose_seq, i, wanted[i], (), max_width)
            for label in wanted[i]:
                path = out_dir / f"{label.lower().replace(' ', '_')}_f{i:04d}.png"
                cv2.imwrite(str(path), image)
                paths[label] = path
    finally:
        cap.release()
    return paths


def _parse_key(text: str) -> tuple[str, int]:
    label, _, idx = text.rpartition("=")
    if not label or not idx.isdigit():
        raise argparse.ArgumentTypeError(f"expected LABEL=FRAME, got {text!r}")
    return label.replace("_", " ").upper(), int(idx)


def main() -> None:
    """Parse the command line and render an overlay video."""
    parser = argparse.ArgumentParser(description="Render a skeleton overlay video.")
    parser.add_argument("video", help="input video")
    parser.add_argument("--out", help="output mp4 (default: outputs/<name>_overlay.mp4)")
    parser.add_argument("--raw", action="store_true", help="do not smooth the landmarks")
    parser.add_argument("--key", action="append", type=_parse_key, default=[],
                        metavar="LABEL=FRAME", help="label a frame, e.g. --key STANCE=12")
    parser.add_argument("--png-dir", help="also export key frames as PNGs into this folder")
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else Path("outputs") / f"{video.stem}_overlay.mp4"
    seq = extract_pose(video)
    if not args.raw:
        seq = smooth_pose(seq)
    keys = dict(args.key)
    print(f"Detection rate: {seq.detection_rate():.1%}")
    print(f"Wrote {render_overlay(video, seq, out, key_frames=keys)}")
    if args.png_dir and keys:
        for label, path in export_key_frames(video, seq, keys, args.png_dir).items():
            print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
