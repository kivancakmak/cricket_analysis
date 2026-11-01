"""Pipeline tests. Pose extraction is replaced by a synthetic sequence (no model needed)."""
import json
from pathlib import Path

import pytest

from src import pipeline
from src.pipeline import analyze


@pytest.fixture
def patched_pose(monkeypatch, make_sequence):
    """Make ``extract_pose`` return a synthetic batter; the calls are recorded."""
    calls = []
    state = {"seq": make_sequence()}

    def fake_extract(video_path, *args, **kwargs):
        calls.append(video_path)
        return state["seq"]

    monkeypatch.setattr(pipeline, "extract_pose", fake_extract)
    return state, calls


def test_good_clip_is_scored_and_rendered(patched_pose, blank_video, tmp_path):
    _, calls = patched_pose
    stages = []
    result = analyze(blank_video(), "forward_defence", tmp_path / "out", on_stage=stages.append)

    assert result.status == "ok" and result.ok and result.message == ""
    assert 0 <= result.score.total <= 100
    assert set(result.score.sub_scores) == {"Head & Balance", "Footwork", "Body Shape"}
    assert result.orientation.facing == "right" and result.orientation.handedness == "right"
    assert calls                                                       # pose really was extracted
    assert stages == ["Checking the video", "Finding the batter", "Finding the key moments",
                      "Checking the clip quality", "Scoring the shot", "Drawing the skeleton video"]

    overlay = Path(result.outputs["overlay_video"])
    assert overlay.exists() and overlay.stat().st_size > 0
    frames = result.outputs["key_frames"]
    assert set(frames) == {"STANCE", "FRONT FOOT DOWN", "CONTACT"}
    assert all(Path(p).exists() for p in frames.values())
    assert {"probe", "pose", "analysis", "quality", "scoring", "render", "total"} <= set(result.timings)


def test_result_is_json_serialisable(patched_pose, blank_video, tmp_path):
    result = analyze(blank_video(), "front_foot_drive", tmp_path)
    data = json.loads(result.to_json())                               # allow_nan=False inside
    assert data["schema_version"] == 1 and data["status"] == "ok" and data["shot_type"] == "front_foot_drive"
    assert data["score"]["total"] == pytest.approx(result.score.total)
    assert data["score"]["rules_status"].startswith("PLACEHOLDER")
    assert set(data["key_moments"]) >= {"stance", "front_foot_landing", "contact"}
    assert "head_stability" in data["metrics"]["values"]
    assert data["outputs"]["overlay_video"].endswith("overlay.mp4")
    assert data["quality"]["ok"] is True and len(data["quality"]["checks"]) == 6
    assert data == json.loads(json.dumps(result.to_dict()))           # nothing non-JSON hiding in it


def test_render_can_be_skipped(patched_pose, blank_video):
    result = analyze(blank_video(), "forward_defence", render=False)
    assert result.ok and result.outputs == {}
    assert "render" not in result.timings and result.to_dict()["outputs"]["overlay_video"] is None


def test_long_clip_is_rejected_before_any_pose_work(patched_pose, blank_video):
    _, calls = patched_pose
    result = analyze(blank_video(frames=300), "forward_defence")      # 10 s
    assert result.status == "rejected" and "too long" in result.message
    assert result.message.startswith("Please film again.")
    assert calls == [] and result.score is None and result.outputs == {}


def test_quality_failure_gives_friendly_message_and_no_score(patched_pose, blank_video):
    state, _ = patched_pose
    state["seq"].landmarks[:, [27, 28, 31, 32], 1] = 0.999            # feet at the bottom edge
    result = analyze(blank_video(), "forward_defence")
    assert result.status == "rejected" and "feet are visible" in result.message
    assert result.score is None and result.outputs == {}
    data = json.loads(result.to_json())
    assert data["quality"]["ok"] is False and data["score"] is None


def test_unreadable_video_is_an_error_result_not_an_exception(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"this is not a video")
    result = analyze(bad, "forward_defence")
    assert result.status == "error" and "couldn't open" in result.message
    json.loads(result.to_json())


def test_unknown_shot_type_is_a_programming_error(blank_video):
    with pytest.raises(ValueError, match="Unknown shot type"):
        analyze(blank_video(), "cover_drive")


def test_pipeline_has_no_ui_dependency():
    assert "gradio" not in Path(pipeline.__file__).read_text(encoding="utf-8")
