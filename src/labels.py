"""Coach labels: storage for the "Coach labelling" tab and the agreement report.

Labels live in ``labels/labels.csv`` in a *long* format - one row per labelled
item - so it works for any rule set and opens cleanly in Excel or pandas::

    timestamp, clip_id, shot_type, coach_id, kind, item, coach_label, ai_label,
    ai_value, ai_confidence, fps, rules_hash, notes

========  ================================  ==========================  ==========================
kind      item                              coach_label                 ai_label / ai_value
========  ================================  ==========================  ==========================
moment    stance / front_foot_landing /     frame the coach chose       frame the AI chose (blank
          contact                                                       if the AI found none)
rule      rule id from the YAML             ``pass`` or ``fail``        AI verdict (``pass``,
                                                                        ``fail`` or ``skipped``) /
                                                                        the measured value
overall   ``overall``                       score 0-100                 AI total score
========  ================================  ==========================  ==========================

Only things the coach actually labelled are stored (an untouched rule or an
unconfirmed key frame is not a label). Saving again for the same
``(clip_id, shot_type, coach_id)`` replaces that coach's earlier labels for the
clip. ``fps`` (needed to turn frame differences into milliseconds) is filled on
moment rows, and ``rules_hash`` records which version of the rules file the AI
values came from, because they change when coaches edit the rules.

This module has no UI code; ``app.py`` and ``scripts/agreement_report.py`` use it.
"""
from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

from src.pipeline import AnalysisResult
from src.scoring import RULES_DIR, load_rules

LABELS_CSV = Path(__file__).resolve().parent.parent / "labels" / "labels.csv"
LABEL_COLUMNS = ["timestamp", "clip_id", "shot_type", "coach_id", "kind", "item", "coach_label",
                 "ai_label", "ai_value", "ai_confidence", "fps", "rules_hash", "notes"]
MOMENT_NAMES = ("stance", "front_foot_landing", "contact")
MAX_LABELLABLE_FRAMES = 600     # refuse to extract frames for anything longer (about 24 s at 25 fps)
RULE_LABELS = ("pass", "fail")


def clip_id_for(video_path: str | Path) -> str:
    """Stable id from the file name and content, so the same clip always gets the same id."""
    path = Path(video_path)
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)[:40].strip("_")
    return f"{stem or 'clip'}-{digest.hexdigest()[:8]}"


def rules_hash(shot_type: str) -> str:
    """Short fingerprint of a shot's rules file (changes whenever a coach edits it)."""
    return hashlib.sha1((RULES_DIR / f"{shot_type}.yaml").read_bytes()).hexdigest()[:8]


def clean_coach_id(text: str | None) -> str:
    return " ".join((text or "").split())[:40]


# --------------------------------------------------------------------------- what the AI said
def ai_snapshot(result: AnalysisResult, clip_id: str) -> dict:
    """The AI's answers for one clip in plain data (kept in the UI between loading and saving)."""
    moments = {name: None for name in MOMENT_NAMES}
    if result.key_moments is not None:
        for name in MOMENT_NAMES:
            moment = getattr(result.key_moments, name)
            moments[name] = None if moment is None else moment.frame
    # Every rule of the shot is listed, so a coach can label rules even when the AI could not
    # score the clip (those rules show as "skipped").
    scored = {r.rule.id: r for r in result.score.rule_results} if result.score is not None else {}
    verdicts = {"passed": "pass", "failed": "fail"}
    rules = []
    for rule in load_rules(result.shot_type).rules:
        r = scored.get(rule.id)
        rules.append({"id": rule.id, "label": rule.label, "category": rule.category,
                      "status": verdicts.get(r.status, "skipped") if r else "skipped",
                      "value": r.value if r else None, "confidence": r.confidence if r else 0.0,
                      "ideal": rule.ideal_text})
    return {"clip_id": clip_id, "shot_type": result.shot_type, "rules_hash": rules_hash(result.shot_type),
            "fps": result.video_info.get("fps"), "ai_status": result.status, "moments": moments,
            "rules": rules, "total": None if result.score is None else result.score.total,
            "confidence": None if result.score is None else result.score.confidence}


# --------------------------------------------------------------------------- CSV storage
def build_rows(snapshot: dict, coach_id: str, coach_moments: dict[str, int] | None,
               coach_rules: dict[str, str], coach_overall: float | None, notes: str = "",
               timestamp: str | None = None) -> list[dict]:
    """Rows to save. ``coach_moments`` is None unless the coach confirmed the key frames."""
    stamp = timestamp or datetime.now().isoformat(timespec="seconds")
    base = {"timestamp": stamp, "clip_id": snapshot["clip_id"], "shot_type": snapshot["shot_type"],
            "coach_id": coach_id, "rules_hash": snapshot["rules_hash"]}
    rows = []
    for name in MOMENT_NAMES:
        if coach_moments and coach_moments.get(name) is not None:
            ai_frame = snapshot["moments"].get(name)
            rows.append({**base, "kind": "moment", "item": name, "coach_label": int(coach_moments[name]),
                         "ai_label": "" if ai_frame is None else ai_frame, "fps": snapshot.get("fps") or ""})
    for rule in snapshot["rules"]:
        verdict = coach_rules.get(rule["id"])
        if verdict in RULE_LABELS:
            rows.append({**base, "kind": "rule", "item": rule["id"], "coach_label": verdict,
                         "ai_label": rule["status"], "ai_value": "" if rule["value"] is None else rule["value"],
                         "ai_confidence": rule["confidence"] if rule["value"] is not None else ""})
    if coach_overall is not None:
        rows.append({**base, "kind": "overall", "item": "overall", "coach_label": coach_overall,
                     "ai_label": "" if snapshot["total"] is None else snapshot["total"],
                     "ai_confidence": "" if snapshot["confidence"] is None else snapshot["confidence"],
                     "notes": notes.strip()})
    elif notes.strip() and rows:
        rows[-1]["notes"] = notes.strip()
    return rows


def load_labels(path: str | Path | None = None) -> list[dict]:
    """All saved label rows (empty list when the file does not exist yet)."""
    path = Path(path) if path else LABELS_CSV
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_labels(rows: list[dict], path: str | Path | None = None) -> int:
    """Add ``rows``, replacing earlier rows of the same (clip, shot, coach). Returns rows saved.

    Written to a temporary file and moved into place, so a crash cannot leave a
    half-written CSV. Raises ``PermissionError`` if the file is open in Excel.
    """
    if not rows:
        return 0
    path = Path(path) if path else LABELS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    replaced = {(r["clip_id"], r["shot_type"], r["coach_id"]) for r in rows}
    kept = [r for r in load_labels(path) if (r["clip_id"], r["shot_type"], r["coach_id"]) not in replaced]
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept + rows)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return len(rows)


def load_coach_labels(clip_id: str, shot_type: str, coach_id: str, path: str | Path | None = None) -> dict:
    """A coach's earlier labels for a clip, to pre-fill the form: moments, rules, overall, notes."""
    out = {"moments": {}, "rules": {}, "overall": None, "notes": ""}
    for r in load_labels(path):
        if (r["clip_id"], r["shot_type"], r["coach_id"]) != (clip_id, shot_type, coach_id):
            continue
        if r["kind"] == "moment" and r["coach_label"] != "":
            out["moments"][r["item"]] = int(float(r["coach_label"]))
        elif r["kind"] == "rule":
            out["rules"][r["item"]] = r["coach_label"]
        elif r["kind"] == "overall" and r["coach_label"] != "":
            out["overall"] = float(r["coach_label"])
        if r.get("notes"):
            out["notes"] = r["notes"]
    return out


# --------------------------------------------------------------------------- frames for the slider
def extract_frames(video_path: str | Path, out_dir: str | Path, max_width: int = 960) -> list[Path]:
    """Save every frame as a JPEG (read sequentially, so frame numbers match the analysis exactly)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    paths: list[Path] = []
    try:
        while len(paths) < MAX_LABELLABLE_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
            path = out_dir / f"{len(paths):04d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            paths.append(path)
    finally:
        cap.release()
    return paths
