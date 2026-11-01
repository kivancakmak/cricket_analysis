"""Stage 1: video -> PoseSequence.

Runs the MediaPipe Pose Landmarker (Tasks API, VIDEO mode) over every frame and
returns the raw landmarks. Smoothing is a separate function so the raw data is
always kept.

CLI:
    python -m src.pose data/sample.mp4
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"
NUM_LANDMARKS = 33

# Landmark array layout: (n_frames, 33, 4) with channels [x, y, z, visibility].
# x, y are normalized to [0, 1] by frame width/height; z is relative depth.
# World landmarks are metric (metres, hip-centred): [x, y, z, visibility].


@dataclass
class PoseSequence:
    """Per-frame pose landmarks for one video.

    ``landmarks`` and ``world_landmarks`` have shape (n_frames, 33, 4); frames
    where no person was detected are NaN and flagged False in ``detected``.
    """

    fps: float
    width: int
    height: int
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    detected: np.ndarray  # (n_frames,) bool
    timestamps_ms: np.ndarray  # (n_frames,) int

    @property
    def n_frames(self) -> int:
        return len(self.detected)

    def frame(self, i: int) -> np.ndarray | None:
        """Landmarks (33, 4) of frame ``i``, or None if no person was detected."""
        return self.landmarks[i] if self.detected[i] else None

    def world_frame(self, i: int) -> np.ndarray | None:
        return self.world_landmarks[i] if self.detected[i] else None

    def detection_rate(self) -> float:
        return float(self.detected.mean()) if self.n_frames else 0.0

    def mean_visibility(self) -> float:
        """Mean landmark visibility over detected frames (0 if none)."""
        if not self.detected.any():
            return 0.0
        return float(self.landmarks[self.detected][:, :, 3].mean())

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            fps=self.fps,
            size=np.array([self.width, self.height]),
            landmarks=self.landmarks,
            world_landmarks=self.world_landmarks,
            detected=self.detected,
            timestamps_ms=self.timestamps_ms,
        )

    @classmethod
    def load(cls, path: str | Path) -> PoseSequence:
        with np.load(path) as d:
            width, height = (int(v) for v in d["size"])
            return cls(
                fps=float(d["fps"]),
                width=width,
                height=height,
                landmarks=d["landmarks"],
                world_landmarks=d["world_landmarks"],
                detected=d["detected"],
                timestamps_ms=d["timestamps_ms"],
            )


def _to_array(landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32
    )


def extract_pose(
    video_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> PoseSequence:
    """Run the pose landmarker over a video and return raw (unsmoothed) landmarks."""
    # Imported lazily so the smoothing/IO helpers work without mediapipe loaded.
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions, vision

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pose model not found at {model_path}. Run: python scripts/download_model.py"
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or math.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    frames, world, detected, stamps = [], [], [], []
    nan = np.full((NUM_LANDMARKS, 4), np.nan, dtype=np.float32)
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            ts = round(i * 1000.0 / fps)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(Image(ImageFormat.SRGB, rgb), ts)
            if result.pose_landmarks:
                frames.append(_to_array(result.pose_landmarks[0]))
                world.append(_to_array(result.pose_world_landmarks[0]))
                detected.append(True)
            else:
                frames.append(nan)
                world.append(nan)
                detected.append(False)
            stamps.append(ts)
            i += 1
    cap.release()

    return PoseSequence(
        fps=float(fps),
        width=width,
        height=height,
        landmarks=np.array(frames, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 4),
        world_landmarks=np.array(world, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 4),
        detected=np.array(detected, dtype=bool),
        timestamps_ms=np.array(stamps, dtype=np.int64),
    )


class _OneEuro:
    """One-Euro filter (Casiez et al.): low lag when moving, low jitter at rest."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


def _smooth_array(arr: np.ndarray, detected: np.ndarray, dt: float,
                  min_cutoff: float, beta: float) -> np.ndarray:
    """One-Euro smooth x, y, z of a (n, 33, 4) array; visibility is left untouched.

    Each run of consecutive detected frames is filtered independently, so
    nothing is smoothed across a detection gap.
    """
    out = arr.copy()
    filt: _OneEuro | None = None
    for i in range(len(arr)):
        if not detected[i]:
            filt = None
            continue
        if filt is None:
            filt = _OneEuro(min_cutoff, beta)
        out[i, :, :3] = filt(arr[i, :, :3].astype(np.float64), dt)
    return out


def smooth_pose(seq: PoseSequence, min_cutoff: float = 1.5, beta: float = 10.0) -> PoseSequence:
    """Return a smoothed copy of ``seq`` (One-Euro filter); the input is not modified.

    Lower ``min_cutoff`` removes more jitter; higher ``beta`` reduces lag during
    fast movement (important for a bat swing). Defaults suit normalized coordinates.
    """
    dt = 1.0 / seq.fps
    return PoseSequence(
        fps=seq.fps,
        width=seq.width,
        height=seq.height,
        landmarks=_smooth_array(seq.landmarks, seq.detected, dt, min_cutoff, beta),
        world_landmarks=_smooth_array(seq.world_landmarks, seq.detected, dt, min_cutoff, beta),
        detected=seq.detected.copy(),
        timestamps_ms=seq.timestamps_ms.copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose from a video and print stats.")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--save", help="optional .npz path to save the raw PoseSequence")
    args = parser.parse_args()

    seq = extract_pose(args.video)
    print(f"Frames: {seq.n_frames}  FPS: {seq.fps:.1f}  Size: {seq.width}x{seq.height}")
    print(f"Detection rate: {seq.detection_rate():.1%}")
    print(f"Mean visibility (detected frames): {seq.mean_visibility():.3f}")
    if args.save:
        seq.save(args.save)
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
