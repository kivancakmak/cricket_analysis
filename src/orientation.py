"""Stage 4: batter orientation from a side-on pose sequence.

Works out, with a confidence for each answer:

* ``facing``     - which way the batter faces in the image (towards the bowler):
                   "left" or "right" of the *image*.
* ``front_leg``  - which of the batter's own legs is the front leg ("left" or
                   "right" leg, i.e. MediaPipe's left/right landmarks).
* ``handedness`` - "right" or "left" handed, inferred from the front leg
                   (a right-hander's left leg and shoulder point at the bowler).

Each answer is a weighted vote of several simple cues (see ``ORIENTATION_CONFIG``).
A cue's vote is scaled by the visibility of the landmarks it uses, so a cue that
rests on an occluded far-side joint counts for less and lowers the confidence.

Assumptions: static camera, one batter, orthodox stance (front leg is the one
nearer the bowler). Stance cues use the first ``window_s`` seconds; the stride
cues use the whole clip. Pass a smoothed sequence (``smooth_pose``) for best results.

CLI:
    python -m src.orientation data/sample.mp4
"""
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.pose import PoseSequence, extract_pose, smooth_pose

# MediaPipe landmark indices ("left"/"right" are the subject's own sides).
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

# Heuristic thresholds. Distances are in torso lengths (shoulder-mid to hip-mid), so
# they do not depend on how large the batter is in the frame. A cue's vote is
# tanh(value / scale): the scale is the size at which that cue is "clearly" decisive.
ORIENTATION_CONFIG: dict = {
    "window_s": 1.0,             # stance cues use the first this-many seconds
    "min_window_frames": 3,      # fewer detected frames than this -> "unknown"
    "min_visibility": 0.3,       # landmarks below this are ignored on that frame
    "min_confidence": 0.15,      # below this an answer is reported as "unknown"
    "edge_fraction": 0.2,        # start/end share of the clip used for stride displacement
    # name: (scale in torso lengths, weight)
    "facing_cues": {
        "head_profile": (0.10, 2.0),    # nose ahead of the ears
        "nose_vs_shoulders": (0.15, 1.0),
        "toe_vs_heel": (0.15, 0.5),
        "stride": (0.30, 1.5),          # the lead ankle travels towards the bowler
    },
    "front_leg_cues": {
        "ankle_order": (0.20, 1.0),     # front ankle is further towards the bowler
        "hip_order": (0.10, 0.75),
        "shoulder_order": (0.10, 0.75),
        "stride": (0.30, 1.5),          # front ankle travels further than the back one
    },
}


@dataclass(frozen=True)
class Orientation:
    """Orientation result. Confidences are in [0, 1]; low-confidence answers are "unknown"."""

    facing: str               # "left" | "right" (image direction) | "unknown"
    facing_confidence: float
    front_leg: str            # "left" | "right" (batter's own leg) | "unknown"
    front_leg_confidence: float
    handedness: str           # "right" | "left" | "unknown"
    handedness_confidence: float
    cues: dict[str, float] = field(default_factory=dict)  # signed vote of each cue, for debugging

    @property
    def back_leg(self) -> str:
        """The leg that is not the front leg ("unknown" if the front leg is unknown)."""
        return {"left": "right", "right": "left"}.get(self.front_leg, "unknown")

    @property
    def confidence(self) -> float:
        """Overall confidence: the weakest of the three answers."""
        return min(self.facing_confidence, self.front_leg_confidence, self.handedness_confidence)


def _points(seq: PoseSequence, frames: np.ndarray, idx: int, min_vis: float) -> np.ndarray:
    """Pixel (x, y) of one landmark over ``frames``, NaN where not visible enough."""
    lm = seq.landmarks[frames, idx]
    pts = lm[:, :2] * (seq.width, seq.height)
    return np.where((lm[:, 3] >= min_vis)[:, None], pts, np.nan)


def _visibility(seq: PoseSequence, frames: np.ndarray, indices: tuple[int, ...]) -> float:
    """Mean visibility of the given landmarks over ``frames``, in [0, 1]."""
    vis = seq.landmarks[frames][:, indices, 3]
    return float(np.clip(np.nanmean(vis), 0.0, 1.0)) if vis.size else 0.0


def _median(values: np.ndarray) -> float:
    """Median ignoring NaN; NaN (no warning) if everything is missing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(values)) if np.isfinite(values).any() else math.nan


def _vote(value: float, scale: float) -> float:
    """Squash a distance (in torso lengths) to a vote in [-1, 1]; NaN stays NaN."""
    return math.tanh(value / scale) if math.isfinite(value) else math.nan


def _combine(cues: list[tuple[float, float, float]]) -> float:
    """Weighted mean vote in [-1, 1] from ``(vote, visibility, weight)`` tuples.

    Missing cues (NaN) add nothing but their weight still counts in the
    denominator, so an answer built from few cues is less confident.
    """
    total = sum(w for _, _, w in cues)
    num = sum(v * vis * w for v, vis, w in cues if math.isfinite(v))
    return num / total if total else 0.0


def _stride_displacement(seq: PoseSequence, idx: int, cfg: dict) -> float:
    """Net horizontal ankle movement (pixels) from the start to the end of the clip."""
    frames = np.flatnonzero(seq.detected)
    k = max(1, round(len(frames) * cfg["edge_fraction"]))
    x = _points(seq, frames, idx, cfg["min_visibility"])[:, 0]
    return _median(x[-k:]) - _median(x[:k])


def detect_orientation(seq: PoseSequence, config: dict | None = None) -> Orientation:
    """Determine facing direction, front/back leg and handedness of the batter."""
    cfg = ORIENTATION_CONFIG if config is None else config
    unknown = Orientation("unknown", 0.0, "unknown", 0.0, "unknown", 0.0)

    window = np.arange(min(seq.n_frames, max(1, round(cfg["window_s"] * seq.fps))))
    window = window[seq.detected[window]]
    if len(window) < cfg["min_window_frames"]:
        return unknown

    mv = cfg["min_visibility"]

    def x_of(idx: int) -> np.ndarray:
        """Pixel x of a landmark over the stance window."""
        return _points(seq, window, idx, mv)[:, 0]

    # Body-size scale: shoulder-mid to hip-mid distance in pixels.
    shoulder_mid = (_points(seq, window, L_SHOULDER, mv) + _points(seq, window, R_SHOULDER, mv)) / 2
    hip_mid = (_points(seq, window, L_HIP, mv) + _points(seq, window, R_HIP, mv)) / 2
    torso = _median(np.linalg.norm(shoulder_mid - hip_mid, axis=1))
    if not math.isfinite(torso) or torso <= 0:
        return unknown

    def offset(a: np.ndarray, b: np.ndarray) -> float:
        """Median horizontal offset a - b, in torso lengths."""
        return _median(a - b) / torso

    with warnings.catch_warnings():  # frames where neither ear is visible give NaN, not a warning
        warnings.simplefilter("ignore", RuntimeWarning)
        ears_x = np.nanmean(np.stack([x_of(L_EAR), x_of(R_EAR)]), axis=0)
    shoulders_x = (x_of(L_SHOULDER) + x_of(R_SHOULDER)) / 2
    toe_minus_heel = np.concatenate([x_of(L_TOE) - x_of(L_HEEL), x_of(R_TOE) - x_of(R_HEEL)])

    # Stride displacement of each ankle, in torso lengths (positive = towards image right).
    disp_l = _stride_displacement(seq, L_ANKLE, cfg) / torso
    disp_r = _stride_displacement(seq, R_ANKLE, cfg) / torso
    lead = disp_l if abs(disp_l) >= abs(disp_r) or not math.isfinite(disp_r) else disp_r
    all_frames = np.flatnonzero(seq.detected)

    # --- facing direction: +1 = right of the image, -1 = left --------------------------
    sc = cfg["facing_cues"]
    facing_votes = {
        "head_profile": (_vote(offset(x_of(NOSE), ears_x), sc["head_profile"][0]),
                         _visibility(seq, window, (NOSE, L_EAR, R_EAR))),
        "nose_vs_shoulders": (_vote(offset(x_of(NOSE), shoulders_x), sc["nose_vs_shoulders"][0]),
                              _visibility(seq, window, (NOSE, L_SHOULDER, R_SHOULDER))),
        "toe_vs_heel": (_vote(_median(toe_minus_heel) / torso, sc["toe_vs_heel"][0]),
                        _visibility(seq, window, (L_TOE, L_HEEL, R_TOE, R_HEEL))),
        "stride": (_vote(lead, sc["stride"][0]),
                   _visibility(seq, all_frames, (L_ANKLE, R_ANKLE))),
    }
    facing_score = _combine([(v, vis, sc[name][1]) for name, (v, vis) in facing_votes.items()])
    facing_conf = abs(facing_score)
    if facing_conf < cfg["min_confidence"]:
        return Orientation("unknown", facing_conf, "unknown", 0.0, "unknown", 0.0,
                           {k: v for k, (v, _) in facing_votes.items()})
    d = 1.0 if facing_score > 0 else -1.0

    # --- front leg: positive vote = batter's LEFT leg is the front leg -----------------
    sc = cfg["front_leg_cues"]
    leg_votes = {
        "ankle_order": (_vote(d * offset(x_of(L_ANKLE), x_of(R_ANKLE)), sc["ankle_order"][0]),
                        _visibility(seq, window, (L_ANKLE, R_ANKLE))),
        "hip_order": (_vote(d * offset(x_of(L_HIP), x_of(R_HIP)), sc["hip_order"][0]),
                      _visibility(seq, window, (L_HIP, R_HIP))),
        "shoulder_order": (_vote(d * offset(x_of(L_SHOULDER), x_of(R_SHOULDER)),
                                 sc["shoulder_order"][0]),
                           _visibility(seq, window, (L_SHOULDER, R_SHOULDER))),
        "stride": (_vote(d * (disp_l - disp_r), sc["stride"][0]),
                   _visibility(seq, all_frames, (L_ANKLE, R_ANKLE))),
    }
    leg_score = _combine([(v, vis, sc[name][1]) for name, (v, vis) in leg_votes.items()])
    # A leg answer is only as trustworthy as the facing direction it was measured against.
    leg_conf = min(abs(leg_score), facing_conf)

    cues = {f"facing:{k}": v for k, (v, _) in facing_votes.items()}
    cues.update({f"front_leg:{k}": v for k, (v, _) in leg_votes.items()})
    facing = "right" if d > 0 else "left"
    if leg_conf < cfg["min_confidence"]:
        return Orientation(facing, facing_conf, "unknown", leg_conf, "unknown", leg_conf, cues)

    front_leg = "left" if leg_score > 0 else "right"
    handedness = "right" if front_leg == "left" else "left"   # right-hander leads with the left leg
    return Orientation(facing, facing_conf, front_leg, leg_conf, handedness, leg_conf, cues)


def main() -> None:
    """Parse the command line and print the orientation of the batter in a video."""
    parser = argparse.ArgumentParser(description="Detect batter orientation in a video.")
    parser.add_argument("video", help="path to a video file")
    args = parser.parse_args()

    seq = smooth_pose(extract_pose(Path(args.video)))
    result = detect_orientation(seq)
    print(f"Facing:      {result.facing} of image   (confidence {result.facing_confidence:.2f})")
    print(f"Front leg:   {result.front_leg}   Back leg: {result.back_leg}"
          f"   (confidence {result.front_leg_confidence:.2f})")
    print(f"Handedness:  {result.handedness}-handed   (confidence {result.handedness_confidence:.2f})")
    for name, vote in result.cues.items():
        print(f"  {name:28s} {vote:+.2f}")


if __name__ == "__main__":
    main()
