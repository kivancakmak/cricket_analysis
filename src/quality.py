"""Stage 8: quality gate - is this clip good enough to score?

Run before scoring. If any check fails the app should show the friendly
"please film again" instruction instead of a score. Checks (in the order
their messages are listed):

1. ``duration``        clip is 2-8 s long.
2. ``detection``       a person was found in enough of the frames.
3. ``frame``           no key joint (or foot) sits on the image border: the whole
                       batter stays in frame.
4. ``visibility``      the key joints (head, hips, knees, ankles, wrists) are clearly
                       visible. Side-on clips hide the far limbs, so for each left/right
                       pair the better-visible (near-side) joint is used.
5. ``camera``          the camera is roughly side-on, i.e. square to the pitch:

                       * the shoulders are not spread across (almost) the whole torso
                         height, which means the camera is looking at the batter's chest
                         or back (``max_shoulder_width_ratio``), and
                       * the feet are clearly separated *sideways* in the image at
                         landing (``min_stride_projection``, in body heights). A camera
                         looking down the pitch sees the stride as depth, not as
                         sideways travel, so this catches front-on/behind views that
                         the shoulder ratio cannot (shoulders overlap in those too).

                       On the sample side-on clips the shoulder ratio is 0.4-0.65 and
                       the stride projection 0.8-1.0, which is what the defaults assume.
6. ``key_moments``     orientation and all three key moments were found with enough
                       confidence.

Every check always runs and is kept in ``QualityReport.checks``. Some failures are
only symptoms of another one (a cut-off foot has low visibility; no person means
no key moments), so the instructions shown to the user list *root causes* only:
a failing check is left out when a check it depends on (``DEPENDS_ON``) also
failed. ``QualityReport.message`` is the first root cause, ``instructions`` lists
them all. Thresholds live in ``QUALITY_CONFIG``.

CLI (runs the whole analysis chain on a video and prints the report):
    python -m src.quality data/sample.mp4
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.metrics import Metrics
from src.orientation import Orientation
from src.phases import KeyMoments
from src.pose import PoseSequence

QUALITY_CONFIG: dict = {
    "min_duration_s": 2.0,
    "max_duration_s": 8.0,
    "min_detection_rate": 0.8,            # share of frames with a person
    "min_joint_visibility": 0.5,          # mean near-side visibility, per joint group
    "border_margin": 0.01,                # a joint closer than this to the edge (fraction) is "at the border"
    "max_border_fraction": 0.1,           # ... in more than this share of detected frames -> fail
    "max_shoulder_width_ratio": 0.9,      # shoulder width in the image / torso length
    "min_stride_projection": 0.30,        # sideways foot gap at landing, in body heights
    "min_orientation_confidence": 0.3,
    "min_moment_confidence": 0.25,
}

# Joint groups checked for visibility: name -> (landmark indices, what to tell the user).
JOINT_GROUPS: dict[str, tuple[tuple[int, ...], str]] = {
    "head": ((0, 7, 8), "your head"),
    "hips": ((23, 24), "your hips"),
    "knees": ((25, 26), "your knees"),
    "ankles": ((27, 28), "your feet"),
    "wrists": ((15, 16), "your hands"),
}
# Joints that must stay inside the frame (nose, wrists, hips, knees, ankles, heels, toes).
FRAME_JOINTS = (0, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)

# A check whose failure is probably a consequence of these checks failing.
DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "detection": ("frame",),            # a batter walking out of shot is "not detected" in those frames
    "visibility": ("detection", "frame"),
    "camera": ("detection",),
    "key_moments": ("duration", "detection", "frame", "visibility", "camera"),
}

MOMENT_ADVICE = {
    "stance": "We couldn't see you settle before the shot. Start filming a second before you move.",
    "front_foot_landing": "We couldn't find where your front foot lands. Film one complete shot and hold the camera still.",
    "contact": "We couldn't find the moment of the shot. Film one complete shot, from set-up to follow-through.",
}


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    detail: str                 # what was measured, e.g. "detected in 62% of frames"
    message: str                # friendly instruction; meant to be shown when the check fails
    value: float | None = None


@dataclass(frozen=True)
class QualityReport:
    checks: tuple[QualityCheck, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def problems(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def root_problems(self) -> list[QualityCheck]:
        """Failing checks that are not just a consequence of another failing check."""
        failed = {c.name for c in self.problems}
        return [c for c in self.problems if not failed & set(DEPENDS_ON.get(c.name, ()))]

    @property
    def instructions(self) -> list[str]:
        """Friendly instructions for the root causes, most important first."""
        return [c.message for c in self.root_problems]

    @property
    def message(self) -> str:
        """Single message for the user: empty when the clip is fine."""
        if self.ok:
            return ""
        return f"Please film again. {self.root_problems[0].message}"


def _check_duration(seq: PoseSequence, cfg: dict) -> QualityCheck:
    seconds = seq.n_frames / seq.fps
    lo, hi = cfg["min_duration_s"], cfg["max_duration_s"]
    if seconds < lo:
        msg = (f"The clip is too short ({seconds:.1f} s). Film the whole shot, from your stance "
               "to the follow-through (about 3-6 seconds).")
    elif seconds > hi:
        msg = f"The clip is too long ({seconds:.0f} s). Trim it to a single shot of about 3-6 seconds."
    else:
        msg = ""
    return QualityCheck("duration", not msg, f"{seconds:.1f} s long (allowed {lo:g}-{hi:g} s)", msg, seconds)


def _check_detection(seq: PoseSequence, cfg: dict) -> QualityCheck:
    rate = seq.detection_rate()
    ok = rate >= cfg["min_detection_rate"]
    msg = "" if ok else (f"We could only see you in {rate:.0%} of the video. Film in good light, with only "
                         "the batter in view and nothing blocking the camera.")
    return QualityCheck("detection", ok, f"person found in {rate:.0%} of frames "
                        f"(need {cfg['min_detection_rate']:.0%})", msg, rate)


def _check_visibility(seq: PoseSequence, cfg: dict) -> QualityCheck:
    if not seq.detected.any():
        return QualityCheck("visibility", True, "not evaluated (no person detected)", "")
    lm = seq.landmarks[seq.detected]
    scores = {name: float(np.mean(lm[:, list(idx), 3].max(axis=1))) for name, (idx, _) in JOINT_GROUPS.items()}
    worst = min(scores, key=scores.get)
    ok = scores[worst] >= cfg["min_joint_visibility"]
    detail = ", ".join(f"{n} {v:.2f}" for n, v in scores.items()) + f" (need {cfg['min_joint_visibility']:g})"
    msg = "" if ok else (f"We couldn't see {JOINT_GROUPS[worst][1]} clearly. Film in good light, in front of a "
                         "plain background, and make sure nothing hides you.")
    return QualityCheck("visibility", ok, detail, msg, scores[worst])


def _check_frame(seq: PoseSequence, cfg: dict) -> QualityCheck:
    if not seq.detected.any():
        return QualityCheck("frame", True, "not evaluated (no person detected)", "")
    xy = seq.landmarks[seq.detected][:, FRAME_JOINTS, :2]
    m = cfg["border_margin"]
    sides = {
        "left": (xy[..., 0] < m).any(axis=1),
        "right": (xy[..., 0] > 1 - m).any(axis=1),
        "top": (xy[..., 1] < m).any(axis=1),
        "bottom": (xy[..., 1] > 1 - m).any(axis=1),
    }
    any_side = np.any(list(sides.values()), axis=0)
    fraction = float(any_side.mean())
    ok = fraction <= cfg["max_border_fraction"]
    side = max(sides, key=lambda s: sides[s].sum())
    advice = {
        "bottom": "Move the camera back so your feet are visible for the whole shot.",
        "top": "Move the camera back, or tilt it up, so your head stays in the frame.",
        "left": "Move the camera back, or keep it on the batter, so you stay inside the frame for the whole shot.",
        "right": "Move the camera back, or keep it on the batter, so you stay inside the frame for the whole shot.",
    }[side]
    detail = (f"a key joint is at the {side} edge in {fraction:.0%} of frames "
              f"(allowed {cfg['max_border_fraction']:.0%})" if fraction else "batter fully inside the frame")
    return QualityCheck("frame", ok, detail, "" if ok else advice, fraction)


def _check_camera(seq: PoseSequence, metrics: Metrics | None, cfg: dict) -> QualityCheck:
    advice = ("The camera should be side-on: place it level with the batter, looking straight across the "
              "pitch (not from in front of or behind the batter).")
    parts, failed = [], []
    value = None
    if seq.detected.any():
        px = seq.landmarks[seq.detected][:, :, :2] * (seq.width, seq.height)
        torso = np.linalg.norm((px[:, 11] + px[:, 12]) / 2 - (px[:, 23] + px[:, 24]) / 2, axis=1)
        ratio = float(np.median(np.abs(px[:, 11, 0] - px[:, 12, 0]) / np.maximum(torso, 1e-6)))
        value = ratio
        parts.append(f"shoulder width {ratio:.2f} of torso length (max {cfg['max_shoulder_width_ratio']:g})")
        if ratio > cfg["max_shoulder_width_ratio"]:
            failed.append("shoulders")
    stride = metrics.get("stride_length_landing") if metrics else None
    if stride is not None:
        parts.append(f"sideways foot gap {abs(stride.value):.2f} body heights (min {cfg['min_stride_projection']:g})")
        if abs(stride.value) < cfg["min_stride_projection"]:
            failed.append("stride")
    if not parts:
        return QualityCheck("camera", True, "not evaluated (no person detected)", "")
    return QualityCheck("camera", not failed, "; ".join(parts), "" if not failed else advice, value)


def _check_key_moments(orientation: Orientation, moments: KeyMoments | None, cfg: dict) -> QualityCheck:
    need = cfg["min_moment_confidence"]
    if min(orientation.facing_confidence, orientation.front_leg_confidence) < cfg["min_orientation_confidence"] \
            or "unknown" in (orientation.facing, orientation.front_leg):
        return QualityCheck(
            "key_moments", False,
            f"orientation unclear (facing {orientation.facing_confidence:.2f}, front leg "
            f"{orientation.front_leg_confidence:.2f}, need {cfg['min_orientation_confidence']:g})",
            "We couldn't tell which way you're facing. Film from the side with only one batter in view.")
    found = {"stance": moments.stance, "front_foot_landing": moments.front_foot_landing,
             "contact": moments.contact} if moments else {}
    problems, detail = [], []
    for name in MOMENT_ADVICE:
        m = found.get(name)
        detail.append(f"{name} {'missing' if m is None else f'{m.confidence:.2f}'}")
        if m is None or m.confidence < need:
            problems.append(name)
    msg = "" if not problems else MOMENT_ADVICE[problems[0]]
    value = min((m.confidence for m in found.values() if m), default=0.0)
    return QualityCheck("key_moments", not problems, ", ".join(detail) + f" (need {need:g})", msg, value)


def assess_quality(seq: PoseSequence, orientation: Orientation, moments: KeyMoments | None,
                   metrics: Metrics | None = None, config: dict | None = None) -> QualityReport:
    """Run every quality check and return the report (see the module docstring)."""
    cfg = QUALITY_CONFIG if config is None else config
    checks = (
        _check_duration(seq, cfg),
        _check_detection(seq, cfg),
        _check_frame(seq, cfg),
        _check_visibility(seq, cfg),
        _check_camera(seq, metrics, cfg),
        _check_key_moments(orientation, moments, cfg),
    )
    return QualityReport(checks)


def main() -> None:
    from src.metrics import compute_metrics
    from src.orientation import detect_orientation
    from src.phases import detect_key_moments
    from src.pose import extract_pose, smooth_pose

    parser = argparse.ArgumentParser(description="Check whether a video is good enough to analyse.")
    parser.add_argument("video", help="path to a video file")
    args = parser.parse_args()

    seq = smooth_pose(extract_pose(Path(args.video)))
    orientation = detect_orientation(seq)
    moments = detect_key_moments(seq, orientation)
    metrics = compute_metrics(seq, orientation, moments)
    report = assess_quality(seq, orientation, moments, metrics)
    for c in report.checks:
        print(f"  [{'ok' if c.passed else 'FAIL'}] {c.name:12s} {c.detail}")
    print(report.message if not report.ok else "Clip is good enough to analyse.")
    for extra in report.instructions[1:]:
        print(f"  also: {extra}")


if __name__ == "__main__":
    main()
