"""Label storage tests: CSV format, upsert behaviour, snapshots of what the AI said."""
import csv

import cv2
import pytest

from src import labels, pipeline
from src.pipeline import analyze

SNAPSHOT = {
    "clip_id": "clip-abc12345", "shot_type": "forward_defence", "rules_hash": "deadbeef", "fps": 25.0,
    "ai_status": "ok", "total": 81.5, "confidence": 0.8,
    "moments": {"stance": 10, "front_foot_landing": 40, "contact": None},
    "rules": [
        {"id": "r1", "label": "Rule 1", "category": "Footwork", "status": "pass", "value": 0.5,
         "confidence": 0.9, "ideal": "0.4 to 0.6"},
        {"id": "r2", "label": "Rule 2", "category": "Footwork", "status": "fail", "value": 1.2,
         "confidence": 0.7, "ideal": "at most 1"},
        {"id": "r3", "label": "Rule 3", "category": "Body Shape", "status": "skipped", "value": None,
         "confidence": 0.0, "ideal": "at least 5"},
    ],
}


def rows_for(coach="alex", moments=None, rules=None, overall=None, notes="", snapshot=SNAPSHOT):
    return labels.build_rows(snapshot, coach, moments, rules or {}, overall, notes,
                             timestamp="2026-01-01T10:00:00")


def test_clip_id_depends_on_content_and_is_filename_safe(tmp_path):
    a, b = tmp_path / "My clip (1).mp4", tmp_path / "other.mp4"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    assert labels.clip_id_for(a) == labels.clip_id_for(a)
    assert labels.clip_id_for(a).startswith("My_clip__1")
    assert labels.clip_id_for(a).split("-")[-1] == labels.clip_id_for(b).split("-")[-1]   # same content, same hash
    b.write_bytes(b"different")
    assert labels.clip_id_for(a).split("-")[-1] != labels.clip_id_for(b).split("-")[-1]


def test_only_what_the_coach_labelled_is_stored():
    rows = rows_for(moments={"stance": 12, "front_foot_landing": 41, "contact": 55},
                    rules={"r1": "pass", "r2": "fail", "r3": "skip"}, overall=70, notes="  nice  ")
    by_item = {r["item"]: r for r in rows}
    assert set(by_item) == {"stance", "front_foot_landing", "contact", "r1", "r2", "overall"}   # r3 skipped
    assert by_item["stance"]["coach_label"] == 12 and by_item["stance"]["ai_label"] == 10
    assert by_item["stance"]["fps"] == 25.0
    assert by_item["contact"]["ai_label"] == ""                      # the AI found no contact
    assert (by_item["r1"]["coach_label"], by_item["r1"]["ai_label"], by_item["r1"]["ai_value"]) == ("pass", "pass", 0.5)
    assert by_item["r2"]["ai_label"] == "fail"
    assert (by_item["overall"]["coach_label"], by_item["overall"]["ai_label"]) == (70, 81.5)
    assert by_item["overall"]["notes"] == "nice"
    assert all(r["rules_hash"] == "deadbeef" and r["coach_id"] == "alex" for r in rows)


def test_unconfirmed_key_frames_and_empty_answers_store_nothing():
    assert rows_for(moments=None, rules={"r1": "skip"}) == []
    assert [r["item"] for r in rows_for(moments=None, overall=50)] == ["overall"]


def test_ai_skipped_rule_keeps_a_blank_value():
    row = rows_for(rules={"r3": "fail"})[0]
    assert row["ai_label"] == "skipped" and row["ai_value"] == ""


def test_save_replaces_the_same_coach_and_keeps_others(tmp_path):
    path = tmp_path / "labels" / "labels.csv"
    assert labels.save_labels(rows_for("alex", rules={"r1": "pass", "r2": "fail"}), path) == 2
    labels.save_labels(rows_for("sam", rules={"r1": "fail"}), path)
    labels.save_labels(rows_for("alex", rules={"r1": "fail"}), path)          # alex relabels

    rows = labels.load_labels(path)
    assert sorted((r["coach_id"], r["item"], r["coach_label"]) for r in rows) == [
        ("alex", "r1", "fail"), ("sam", "r1", "fail")]
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == labels.LABEL_COLUMNS
    assert [p.name for p in path.parent.iterdir()] == ["labels.csv"]           # no temp files left behind


def test_save_nothing_does_not_create_the_file(tmp_path):
    assert labels.save_labels([], tmp_path / "labels.csv") == 0
    assert not (tmp_path / "labels.csv").exists() and labels.load_labels(tmp_path / "labels.csv") == []


def test_failed_save_leaves_the_existing_file_untouched(tmp_path, monkeypatch):
    path = tmp_path / "labels.csv"
    labels.save_labels(rows_for(rules={"r1": "pass"}), path)
    before = path.read_bytes()

    def locked(*args, **kwargs):
        raise PermissionError("locked by Excel")

    monkeypatch.setattr(labels.os, "replace", locked)
    with pytest.raises(PermissionError):
        labels.save_labels(rows_for(rules={"r1": "fail"}), path)
    assert path.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == ["labels.csv"]


def test_coach_labels_round_trip_for_prefilling(tmp_path):
    path = tmp_path / "labels.csv"
    labels.save_labels(rows_for(moments={"stance": 12, "front_foot_landing": 41, "contact": 55},
                                rules={"r1": "pass", "r2": "fail"}, overall=72, notes="good"), path)
    got = labels.load_coach_labels("clip-abc12345", "forward_defence", "alex", path)
    assert got == {"moments": {"stance": 12, "front_foot_landing": 41, "contact": 55},
                   "rules": {"r1": "pass", "r2": "fail"}, "overall": 72.0, "notes": "good"}
    assert labels.load_coach_labels("clip-abc12345", "forward_defence", "someone_else", path)["rules"] == {}


def test_extract_frames_is_sequential_and_resized(blank_video, tmp_path):
    frames = labels.extract_frames(blank_video(frames=20), tmp_path / "f", max_width=160)
    assert len(frames) == 20 and frames[0].name == "0000.jpg" and frames[-1].name == "0019.jpg"
    assert cv2.imread(str(frames[0])).shape[:2] == (120, 160)         # 320x240 scaled to width 160


def test_snapshot_lists_every_rule_even_when_the_clip_was_rejected(blank_video):
    rejected = analyze(blank_video(frames=300), "forward_defence")
    snap = labels.ai_snapshot(rejected, "x-1")
    assert snap["ai_status"] == "rejected" and snap["total"] is None
    assert len(snap["rules"]) == 9 and {r["status"] for r in snap["rules"]} == {"skipped"}
    assert snap["moments"] == {"stance": None, "front_foot_landing": None, "contact": None}


def test_snapshot_of_a_scored_clip(monkeypatch, make_sequence, blank_video):
    monkeypatch.setattr(pipeline, "extract_pose", lambda *a, **k: make_sequence())
    result = analyze(blank_video(), "front_foot_drive", render=False)
    snap = labels.ai_snapshot(result, "x-2")
    assert snap["ai_status"] == "ok" and snap["total"] == pytest.approx(result.score.total)
    assert all(isinstance(v, int) for v in snap["moments"].values())
    assert {r["status"] for r in snap["rules"]} <= {"pass", "fail", "skipped"} and snap["fps"] == 30.0
    assert snap["rules_hash"] == labels.rules_hash("front_foot_drive")
