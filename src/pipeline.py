"""Stage 9: the whole analysis behind one function.

    result = analyze("clip.mp4", "forward_defence")

Stages: clip-length pre-check -> pose -> orientation -> key moments -> metrics ->
quality gate -> scoring -> overlay video and key-frame images.

This module knows nothing about Gradio or any UI. Everything it returns is plain
data: ``AnalysisResult.to_dict()`` / ``to_json()`` give a JSON-safe description
(no NaN, no numpy), so the same function can sit behind a cloud API or be ported
on-device later. Files it writes (overlay video, key-frame PNGs) are returned as
paths in ``result.outputs``.

Outcomes (``result.status``):

* ``"ok"``        - analysed and scored.
* ``"rejected"``  - the quality gate failed; ``result.message`` is the friendly
                    "please film again" text and nothing is scored.
* ``"error"``     - the video could not be read or the pose model is missing.

Programmer errors (unknown shot type, a malformed rules file) raise instead.

CLI:
    python -m src.pipeline data/sample.mp4 forward_defence [--out outputs/run1]
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.metrics import Metrics, compute_metrics
from src.orientation import Orientation, detect_orientation
from src.overlay import export_key_frames, render_overlay
from src.phases import KeyMoment, KeyMoments, detect_key_moments
from src.pose import extract_pose, smooth_pose
from src.quality import QualityCheck, QualityReport, assess_quality, check_clip_length
from src.scoring import SHOT_TYPES, ScoreResult, load_rules, score_shot

SCHEMA_VERSION = 1
SHOT_LABELS = {"forward_defence": "Forward defence", "front_foot_drive": "Front-foot drive"}

# Metrics shown as short notes on the overlay video: moment -> [(metric, text format)].
OVERLAY_NOTES: dict[str, list[tuple[str, str]]] = {
    "front_foot_landing": [("stride_length_landing", "Stride {:.2f} body heights"),
                           ("front_knee_angle_landing", "Front knee {:.0f} deg")],
    "contact": [("head_over_front_knee_contact", "Head over knee {:+.2f}"),
                ("front_elbow_height_contact", "Front elbow {:.2f}")],
}


@dataclass
class AnalysisResult:
    """Everything the analysis produced. Use ``to_dict()`` for a JSON-safe view."""

    status: str                                   # "ok" | "rejected" | "error"
    shot_type: str
    message: str = ""                             # friendly text when rejected / error
    video_info: dict = field(default_factory=dict)
    quality: QualityReport | None = None
    orientation: Orientation | None = None
    key_moments: KeyMoments | None = None
    metrics: Metrics | None = None
    score: ScoreResult | None = None
    outputs: dict = field(default_factory=dict)   # {"overlay_video": path, "key_frames": {label: path}}
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the clip was analysed and scored."""
        return self.status == "ok"

    def to_dict(self) -> dict:
        """JSON-safe dictionary of the whole result (NaN becomes null)."""
        return _clean({
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "shot_type": self.shot_type,
            "message": self.message,
            "video": self.video_info,
            "quality": _quality_dict(self.quality),
            "orientation": _orientation_dict(self.orientation),
            "key_moments": _moments_dict(self.key_moments),
            "metrics": _metrics_dict(self.metrics),
            "score": _score_dict(self.score),
            "outputs": {
                "overlay_video": _path(self.outputs.get("overlay_video")),
                "key_frames": {k: _path(v) for k, v in self.outputs.get("key_frames", {}).items()},
            },
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        })

    def to_json(self, indent: int | None = 2) -> str:
        """``to_dict()`` as a JSON string (never contains NaN)."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


# --------------------------------------------------------------------------- serialisation
def _path(p: object | None) -> str | None:
    return None if p is None else str(p)


def _clean(obj: object) -> object:
    """Recursively make plain JSON data: floats stay floats, NaN/inf become None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):                      # numpy scalar
        return _clean(obj.item())
    return obj


def _check_dict(c: QualityCheck) -> dict:
    return {"name": c.name, "passed": c.passed, "detail": c.detail, "message": c.message, "value": c.value}


def _quality_dict(q: QualityReport | None) -> dict | None:
    if q is None:
        return None
    return {"ok": q.ok, "message": q.message, "instructions": q.instructions,
            "checks": [_check_dict(c) for c in q.checks]}


def _orientation_dict(o: Orientation | None) -> dict | None:
    if o is None:
        return None
    return {"facing": o.facing, "facing_confidence": o.facing_confidence,
            "front_leg": o.front_leg, "back_leg": o.back_leg, "front_leg_confidence": o.front_leg_confidence,
            "handedness": o.handedness, "handedness_confidence": o.handedness_confidence,
            "confidence": o.confidence}


def _moments_dict(k: KeyMoments | None) -> dict | None:
    if k is None:
        return None
    def one(m: KeyMoment | None) -> dict | None:
        """One key moment as plain data, or None."""
        return None if m is None else {"frame": m.frame, "time_s": m.time_s, "confidence": m.confidence}

    return {"stance": one(k.stance), "front_foot_landing": one(k.front_foot_landing),
            "contact": one(k.contact), "notes": list(k.notes)}


def _metrics_dict(m: Metrics | None) -> dict | None:
    if m is None:
        return None
    return {"values": {n: {"value": v.value, "confidence": v.confidence} for n, v in m.values.items()},
            "torso_px": m.torso_px, "body_height_px": m.body_height_px, "notes": list(m.notes)}


def _score_dict(s: ScoreResult | None) -> dict | None:
    if s is None:
        return None
    return {
        "total": s.total, "confidence": s.confidence, "rules_status": s.status, "title": s.title,
        "sub_scores": {n: {"score": x.score, "confidence": x.confidence,
                           "rules_scored": x.rules_scored, "rules_total": x.rules_total}
                       for n, x in s.sub_scores.items()},
        "rules": [{"id": r.rule.id, "label": r.rule.label, "category": r.rule.category,
                   "status": r.status, "value": r.value, "ideal": r.rule.ideal_text,
                   "score": r.score, "weight": r.rule.weight, "confidence": r.confidence,
                   "too": r.side} for r in s.rule_results],
        "fixes": [{"rule_id": f.rule_id, "label": f.label, "category": f.category, "fix": f.text,
                   "drill": f.drill, "value": f.value, "ideal": f.ideal_text, "priority": f.priority}
                  for f in s.fixes],
    }


# --------------------------------------------------------------------------- the pipeline
def probe_video(video_path: str | Path) -> dict:
    """Basic facts about a video; ``duration_s`` is None if the container does not say."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if fps > 0 and frames > 0 else None
        return {"fps": fps, "frames": frames, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "duration_s": duration}
    finally:
        cap.release()


def _overlay_annotations(moments: KeyMoments, metrics: Metrics) -> dict[int, list[str]]:
    notes: dict[int, list[str]] = {}
    for moment_name, items in OVERLAY_NOTES.items():
        moment = getattr(moments, moment_name)
        if moment is None:
            continue
        for metric, fmt in items:
            m = metrics.get(metric)
            if m is not None:
                notes.setdefault(moment.frame, []).append(fmt.format(m.value))
    return notes


def analyze(
    video_path: str | Path,
    shot_type: str,
    output_dir: str | Path | None = None,
    *,
    render: bool = True,
    on_stage: Callable[[str], None] | None = None,
) -> AnalysisResult:
    """Analyse one batting clip.

    Args:
        video_path: the clip (any format OpenCV can read).
        shot_type: ``"forward_defence"`` or ``"front_foot_drive"``.
        output_dir: where to write the overlay video and key-frame PNGs
            (default: a fresh temporary folder).
        render: set False to skip the overlay/PNG step (faster; for batch evaluation).
        on_stage: optional callback told the name of each stage as it starts
            (lets a UI show progress without the pipeline depending on it).
    """
    if shot_type not in SHOT_TYPES:
        raise ValueError(f"Unknown shot type {shot_type!r}; choose one of {list(SHOT_TYPES)}")
    ruleset = load_rules(shot_type)            # fail fast on a broken rules file
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def stage(name: str) -> float:
        """Tell the UI a stage is starting and return the start time."""
        if on_stage:
            on_stage(name)
        return time.perf_counter()

    def finish(result: AnalysisResult) -> AnalysisResult:
        """Record the total time and return the result."""
        timings["total"] = time.perf_counter() - started
        result.timings = timings
        return result

    # 0. Cheap pre-check: refuse clips of the wrong length before any heavy work.
    t = stage("Checking the video")
    try:
        info = probe_video(video_path)
    except ValueError:
        return finish(AnalysisResult("error", shot_type, "We couldn't open that video. "
                                     "Try an .mp4 file filmed on a phone."))
    timings["probe"] = time.perf_counter() - t
    if info["duration_s"] is not None:
        length = check_clip_length(info["duration_s"])
        if not length.passed:
            report = QualityReport((length,))
            return finish(AnalysisResult("rejected", shot_type, report.message, info, report))

    # 1. Pose.
    t = stage("Finding the batter")
    try:
        raw = extract_pose(video_path)
    except FileNotFoundError as exc:
        return finish(AnalysisResult("error", shot_type, f"The pose model is missing. {exc}", info))
    except ValueError:
        return finish(AnalysisResult("error", shot_type, "We couldn't read that video. "
                                     "Try an .mp4 file filmed on a phone.", info))
    seq = smooth_pose(raw)
    info = {**info, "fps": seq.fps, "frames": seq.n_frames, "width": seq.width, "height": seq.height,
            "duration_s": seq.n_frames / seq.fps}
    timings["pose"] = time.perf_counter() - t

    # 2-4. Orientation, key moments, metrics.
    t = stage("Finding the key moments")
    orientation = detect_orientation(seq)
    moments = detect_key_moments(seq, orientation)
    metrics = compute_metrics(seq, orientation, moments)
    timings["analysis"] = time.perf_counter() - t

    # 5. Quality gate.
    t = stage("Checking the clip quality")
    report = assess_quality(seq, orientation, moments, metrics)
    timings["quality"] = time.perf_counter() - t
    if not report.ok:
        return finish(AnalysisResult("rejected", shot_type, report.message, info, report,
                                     orientation, moments, metrics))

    # 6. Scoring.
    t = stage("Scoring the shot")
    score = score_shot(metrics, ruleset)
    timings["scoring"] = time.perf_counter() - t
    result = AnalysisResult("ok", shot_type, "", info, report, orientation, moments, metrics, score)

    # 7. Overlay video and key-frame images.
    if render:
        t = stage("Drawing the skeleton video")
        out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="cricket_"))
        key_frames = moments.as_key_frames()
        overlay = render_overlay(video_path, seq, out / "overlay.mp4", key_frames=key_frames,
                                 annotations=_overlay_annotations(moments, metrics))
        images = export_key_frames(video_path, seq, key_frames, out / "key_frames")
        result.outputs = {"overlay_video": overlay, "key_frames": images}
        timings["render"] = time.perf_counter() - t
    return finish(result)


def main() -> None:
    """Parse the command line, analyse the clip and print the JSON result."""
    parser = argparse.ArgumentParser(description="Analyse a batting clip and print the result as JSON.")
    parser.add_argument("video")
    parser.add_argument("shot_type", choices=SHOT_TYPES)
    parser.add_argument("--out", help="folder for the overlay video and key frames")
    parser.add_argument("--no-render", action="store_true", help="skip the overlay video")
    args = parser.parse_args()
    print(analyze(args.video, args.shot_type, args.out, render=not args.no_render).to_json())


if __name__ == "__main__":
    main()
