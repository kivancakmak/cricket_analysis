"""Stage 6: technique metrics measured at the key moments.

Every metric is a ``Metric(value, confidence)`` and is scale-free, so it does not
depend on how far the batter is from the camera:

=============================  =======================================================  =========
metric                         meaning                                                  unit
=============================  =======================================================  =========
head_over_front_knee_{landing,contact}
                               head x minus front-knee x, positive = head ahead of      torso lengths
                               the knee (towards the bowler)
front_knee_angle_landing       hip-knee-ankle angle of the front leg; 180 = straight    degrees
stride_length_landing          front ankle ahead of back ankle, along the direction     body heights
                               of play
balance_{landing,contact}      hip centre between the feet: 0 = over the back foot,     fraction
                               1 = over the front foot (can leave 0..1)
head_stability                 largest head movement between landing and contact        torso lengths
front_elbow_height_contact     front elbow between wrist (0) and shoulder (1); >1 =     fraction
                               elbow above the shoulder
spine_tilt_{landing,contact}   shoulder-mid to hip-mid line vs vertical; positive =     degrees
                               leaning towards the bowler
=============================  =======================================================  =========

Conventions
-----------
* "Torso length" is the shoulder-mid to hip-mid distance and "body height" the
  nose-to-ankle height, in pixels, each taken as an upper percentile over the
  clip (``scale_percentile``, default 90th). A median would be dragged down by
  the long low lunge after the shot and make strides look too big. Horizontal
  and vertical pixel distances use the image's real aspect ratio.
* "Front" side/arm/leg is the batter's front leg from the orientation result
  (a right-hander's left side); "ahead" means towards the bowler, from the
  facing direction.
* A metric is measured on the median pose of the frames around its key moment
  (``pose_window_frames`` either side), which damps single-frame jitter.
* Angles use 2D image coordinates, which is right for a side-on camera.
* The head is the mean of the ears when they are visible, otherwise the nose
  (the nose sits ahead of the head's centre in profile, which would bias
  "head over knee" forward).

Confidence
----------
``min(visibility of every landmark the metric uses) * (0.5 + 0.5 * key-moment
confidence)``: a metric is only as trustworthy as its least visible joint and
as the frame it was measured on. Metrics that cannot be computed (missing key
moment, joint not tracked, degenerate geometry) are left out of
``Metrics.values`` and explained in ``Metrics.notes``.

CLI:
    python -m src.metrics data/sample.mp4
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.orientation import Orientation, detect_orientation
from src.phases import KeyMoment, KeyMoments, detect_key_moments
from src.pose import PoseSequence, extract_pose, smooth_pose

NOSE = 0
L_EAR, R_EAR = 7, 8
LEFT = {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27}
RIGHT = {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28}

METRIC_CONFIG: dict = {
    "pose_window_frames": 1,   # measure on the median pose of key frame +- this many frames
    "scale_percentile": 90,    # body size = this percentile of torso length / height over the clip
    "min_visibility": 0.1,     # a landmark less visible than this makes the metric unavailable
    "ear_visibility": 0.3,     # ears at least this visible are used for the head position
    "min_base_width": 0.15,    # feet closer than this (torso lengths): balance is undefined
    "min_arm_span": 0.10,      # wrist and shoulder closer than this in height: elbow height undefined
}


@dataclass(frozen=True)
class Metric:
    value: float
    confidence: float


@dataclass(frozen=True)
class Metrics:
    values: dict[str, Metric]
    torso_px: float = math.nan
    body_height_px: float = math.nan
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __getitem__(self, name: str) -> Metric:
        return self.values[name]

    def get(self, name: str) -> Metric | None:
        return self.values.get(name)


class _Pose:
    """Median pose (pixel coordinates) and mean visibility around one key frame."""

    def __init__(self, xy: np.ndarray, vis: np.ndarray):
        self.xy, self.vis = xy, vis

    def point(self, idx: int) -> np.ndarray:
        return self.xy[idx]

    def min_vis(self, indices) -> float:
        return float(min(self.vis[i] for i in indices))

    def visible(self, idx: int, threshold: float) -> bool:
        return bool(np.isfinite(self.xy[idx]).all() and self.vis[idx] >= threshold)


@dataclass(frozen=True)
class _Context:
    d: float                  # +1 if facing image-right, -1 if image-left
    front: dict[str, int]     # landmark indices of the front side
    back: dict[str, int]
    torso: float              # px
    height: float             # px
    cfg: dict


class _Unavailable(Exception):
    """Raised by a metric when its geometry is degenerate; the message goes into the notes."""


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at ``b`` between segments b->a and b->c, in degrees."""
    u, v = a - b, c - b
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm == 0:
        raise _Unavailable("zero-length limb segment")
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / norm, -1.0, 1.0))))


def _head(p: _Pose, c: _Context) -> tuple[float, tuple[int, ...]]:
    """Head x and the landmarks it came from: ears if visible, else the nose."""
    ears = tuple(i for i in (L_EAR, R_EAR) if p.visible(i, c.cfg["ear_visibility"]))
    used = ears or (NOSE,)
    return float(np.mean([p.point(i)[0] for i in used])), used


# Each metric function returns (value, landmarks it used) from a pose snapshot.
def _head_over_front_knee(p: _Pose, c: _Context):
    head_x, used = _head(p, c)
    knee = p.point(c.front["knee"])
    return c.d * (head_x - knee[0]) / c.torso, (*used, c.front["knee"])


def _front_knee_angle(p: _Pose, c: _Context):
    idx = (c.front["hip"], c.front["knee"], c.front["ankle"])
    return _angle(*(p.point(i) for i in idx)), idx


def _stride_length(p: _Pose, c: _Context):
    front, back = p.point(c.front["ankle"]), p.point(c.back["ankle"])
    return c.d * (front[0] - back[0]) / c.height, (c.front["ankle"], c.back["ankle"])


def _balance(p: _Pose, c: _Context):
    hips = (c.front["hip"], c.back["hip"])
    hip_x = (p.point(hips[0])[0] + p.point(hips[1])[0]) / 2
    front_x, back_x = p.point(c.front["ankle"])[0], p.point(c.back["ankle"])[0]
    base = c.d * (front_x - back_x)
    if base / c.torso < c.cfg["min_base_width"]:
        raise _Unavailable("feet too close together (or crossed) to define a base")
    return c.d * (hip_x - back_x) / base, (*hips, c.front["ankle"], c.back["ankle"])


def _front_elbow_height(p: _Pose, c: _Context):
    idx = (c.front["shoulder"], c.front["elbow"], c.front["wrist"])
    shoulder_y, elbow_y, wrist_y = (p.point(i)[1] for i in idx)
    span = wrist_y - shoulder_y            # image y grows downwards: wrist below shoulder -> positive
    if span / c.torso < c.cfg["min_arm_span"]:
        raise _Unavailable("wrist is not below the shoulder; elbow height undefined")
    return (wrist_y - elbow_y) / span, idx


def _spine_tilt(p: _Pose, c: _Context):
    idx = (c.front["shoulder"], c.back["shoulder"], c.front["hip"], c.back["hip"])
    shoulders = (p.point(idx[0]) + p.point(idx[1])) / 2
    hips = (p.point(idx[2]) + p.point(idx[3])) / 2
    lean = c.d * (shoulders[0] - hips[0])
    rise = hips[1] - shoulders[1]          # positive when the shoulders are above the hips
    return math.degrees(math.atan2(lean, rise)), idx


# metric name -> (key moment it is measured at, function)
METRIC_SPECS: dict[str, tuple[str, Callable]] = {
    "head_over_front_knee_landing": ("landing", _head_over_front_knee),
    "head_over_front_knee_contact": ("contact", _head_over_front_knee),
    "front_knee_angle_landing": ("landing", _front_knee_angle),
    "stride_length_landing": ("landing", _stride_length),
    "balance_landing": ("landing", _balance),
    "balance_contact": ("contact", _balance),
    "front_elbow_height_contact": ("contact", _front_elbow_height),
    "spine_tilt_landing": ("landing", _spine_tilt),
    "spine_tilt_contact": ("contact", _spine_tilt),
}


#: Every metric name ``compute_metrics`` can produce (the rule files refer to these).
METRIC_NAMES: tuple[str, ...] = (*METRIC_SPECS, "head_stability")


def _snapshot(seq: PoseSequence, frame: int, k: int) -> _Pose | None:
    lo, hi = max(0, frame - k), min(seq.n_frames, frame + k + 1)
    frames = [i for i in range(lo, hi) if seq.detected[i]]
    if not frames:
        return None
    lm = seq.landmarks[frames]
    xy = np.median(lm[:, :, :2] * (seq.width, seq.height), axis=0)
    return _Pose(xy, lm[:, :, 3].mean(axis=0))


def _body_scale(seq: PoseSequence, percentile: float) -> tuple[float, float]:
    """Upright torso length and body height in pixels (NaN if untrackable)."""
    lm = seq.landmarks[seq.detected]
    if len(lm) == 0:
        return math.nan, math.nan
    xy = lm[:, :, :2] * (seq.width, seq.height)
    shoulders = (xy[:, 11] + xy[:, 12]) / 2
    hips = (xy[:, 23] + xy[:, 24]) / 2
    torso = float(np.percentile(np.linalg.norm(shoulders - hips, axis=1), percentile))
    height = float(np.percentile((xy[:, 27, 1] + xy[:, 28, 1]) / 2 - xy[:, NOSE, 1], percentile))
    return torso, height


def _head_stability(seq: PoseSequence, landing: KeyMoment, contact: KeyMoment,
                    c: _Context) -> tuple[float, float]:
    """Largest nose displacement from its landing position up to contact; (value, min visibility)."""
    k = c.cfg["pose_window_frames"]
    ref = _snapshot(seq, landing.frame, k)
    frames = [i for i in range(landing.frame, contact.frame + 1) if seq.detected[i]]
    if ref is None or not frames:
        raise _Unavailable("no detected frames between landing and contact")
    nose = seq.landmarks[frames, NOSE]
    pts = nose[:, :2] * (seq.width, seq.height)
    value = float(np.max(np.linalg.norm(pts - ref.point(NOSE), axis=1))) / c.torso
    return value, float(nose[:, 3].mean())


def _confidence(visibility: float, *moments: KeyMoment) -> float:
    moment_conf = float(np.mean([m.confidence for m in moments]))
    return float(np.clip(visibility * (0.5 + 0.5 * moment_conf), 0.0, 1.0))


def compute_metrics(seq: PoseSequence, orientation: Orientation, moments: KeyMoments,
                    config: dict | None = None) -> Metrics:
    """Measure every metric at its key moment; see the module docstring for definitions."""
    cfg = METRIC_CONFIG if config is None else config
    if orientation.facing == "unknown" or orientation.front_leg == "unknown":
        return Metrics({}, notes=("Orientation unknown; no metrics computed.",))
    torso, height = _body_scale(seq, cfg["scale_percentile"])
    if not (math.isfinite(torso) and math.isfinite(height) and torso > 0 and height > 0):
        return Metrics({}, notes=("Could not estimate the batter's body size.",))

    front_is_left = orientation.front_leg == "left"
    ctx = _Context(
        d=1.0 if orientation.facing == "right" else -1.0,
        front=LEFT if front_is_left else RIGHT,
        back=RIGHT if front_is_left else LEFT,
        torso=torso, height=height, cfg=cfg,
    )
    by_name = {"landing": moments.front_foot_landing, "contact": moments.contact}
    values: dict[str, Metric] = {}
    notes: list[str] = []

    for name, (moment_name, fn) in METRIC_SPECS.items():
        moment = by_name[moment_name]
        if moment is None:
            notes.append(f"{name}: no {moment_name} moment.")
            continue
        pose = _snapshot(seq, moment.frame, cfg["pose_window_frames"])
        if pose is None:
            notes.append(f"{name}: no person detected near frame {moment.frame}.")
            continue
        try:
            value, used = fn(pose, ctx)
        except _Unavailable as exc:
            notes.append(f"{name}: {exc}.")
            continue
        used = tuple(dict.fromkeys(used))
        if not all(np.isfinite(pose.point(i)).all() for i in used) or not math.isfinite(value) \
                or pose.min_vis(used) < cfg["min_visibility"]:
            notes.append(f"{name}: a required joint was not tracked.")
            continue
        values[name] = Metric(float(value), _confidence(pose.min_vis(used), moment))

    if moments.front_foot_landing is not None and moments.contact is not None:
        try:
            value, vis = _head_stability(seq, moments.front_foot_landing, moments.contact, ctx)
            values["head_stability"] = Metric(
                value, _confidence(vis, moments.front_foot_landing, moments.contact))
        except _Unavailable as exc:
            notes.append(f"head_stability: {exc}.")
    else:
        notes.append("head_stability: needs both landing and contact.")

    return Metrics(values, torso, height, tuple(notes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute technique metrics for a video.")
    parser.add_argument("video", help="path to a video file")
    args = parser.parse_args()

    seq = smooth_pose(extract_pose(Path(args.video)))
    orientation = detect_orientation(seq)
    moments = detect_key_moments(seq, orientation)
    metrics = compute_metrics(seq, orientation, moments)
    print(f"Key moments: {moments.as_key_frames()}")
    for name, m in metrics.values.items():
        print(f"  {name:32s} {m.value:8.3f}   (confidence {m.confidence:.2f})")
    for note in metrics.notes:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
