"""Coach-labelling handlers: load a clip, preview frames, save labels (no browser needed)."""
import pytest

import labelling_tab as tab
from src import labels, pipeline


@pytest.fixture
def csv_path(tmp_path, monkeypatch):
    path = tmp_path / "labels" / "labels.csv"
    monkeypatch.setattr(labels, "LABELS_CSV", path)
    return path


@pytest.fixture
def loaded(monkeypatch, make_sequence, blank_video, csv_path):
    """A clip loaded for coach 'alex' with the synthetic good batter."""
    monkeypatch.setattr(pipeline, "extract_pose", lambda *a, **k: make_sequence())
    video = blank_video()
    values = tab.load_clip(str(video), "forward_defence", "alex")
    return video, values


def test_loading_needs_a_coach_id_and_a_clip(blank_video):
    no_coach = tab.load_clip(str(blank_video()), "forward_defence", "   ")
    assert "coach ID" in no_coach["status"] and no_coach["form"]["visible"] is False
    no_clip = tab.load_clip(None, "forward_defence", "alex")
    assert "add a clip" in no_clip["status"] and "state" not in no_clip


def test_loading_prepares_the_whole_form(loaded):
    _, v = loaded
    state = v["state"]
    assert v["form"]["visible"] is True and "scored" in v["status"] and "alex" in v["status"]
    assert len(state["frames"]) == 90 and state["coach_id"] == "alex" and state["fps"] == 30.0
    assert "score-number" in v["ai_card"]
    for moment in tab.MOMENT_NAMES:
        slider = v[f"{moment}_slider"]
        assert slider["maximum"] == 89 and slider["value"] == state["moments"][moment]   # starts on the AI frame
        assert v[f"{moment}_image"].endswith(".jpg") and "you agree" in v[f"{moment}_info"]
    visible = [v[f"rule_{i}"]["visible"] for i in range(tab.MAX_RULE_SLOTS)]
    assert visible == [True] * 9 + [False] * (tab.MAX_RULE_SLOTS - 9)
    first = v["rule_0"]
    assert first["label"] == "Head over the front knee when the front foot lands"
    assert "target" in first["info"] and first["value"] is None and v["confirm"]["value"] is False


def test_rejected_clip_can_still_be_labelled(blank_video, csv_path):
    values = tab.load_clip(str(blank_video(frames=300)), "forward_defence", "alex")      # 10 s: AI refuses
    assert values["form"]["visible"] is True and "not scored" in values["status"]
    assert all(values[f"{m}_slider"]["value"] == 0 for m in tab.MOMENT_NAMES)
    assert "did not find" in values["stance_info"]
    assert values["rule_0"]["info"].startswith("AI: could not measure")


def test_too_many_frames_and_unreadable_files_are_refused(blank_video, tmp_path, csv_path):
    long = tab.load_clip(str(blank_video(frames=700)), "forward_defence", "alex")
    assert "700 frames" in long["status"] and long["form"]["visible"] is False
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert "couldn't open" in tab.load_clip(str(bad), "forward_defence", "alex")["status"]


def test_frame_preview_describes_the_gap_to_the_ai(loaded):
    _, v = loaded
    ai = v["state"]["moments"]["front_foot_landing"]
    image, info = tab.show_frame(ai - 3, v["state"], "front_foot_landing")
    assert image.endswith(f"{ai - 3:04d}.jpg")
    assert f"the AI chose frame {ai}, 3 frames later (100 ms)" in info
    assert "earlier" in tab.show_frame(ai + 1, v["state"], "front_foot_landing")[1]
    assert tab.show_frame(10_000, v["state"], "contact")[0].endswith("0089.jpg")           # clamped
    assert tab.show_frame(5, None, "contact") == (None, "")


def save(state, **overrides):
    args = {"coach_id": "alex", "confirmed": True, "stance": 12, "landing": 41, "contact": 55,
            "overall": 70, "notes": "", "rules": ["pass", "fail"] + [None] * 14}
    args.update(overrides)
    return tab.save_clip(state, args["coach_id"], args["confirmed"], args["stance"], args["landing"],
                         args["contact"], args["overall"], args["notes"], *args["rules"])


def test_saving_writes_the_labels_and_replaces_earlier_ones(loaded, csv_path):
    _, v = loaded
    message = save(v["state"], notes="late hands")
    assert "Saved 6 labels" in message and "alex" in message and "1 clip in total" in message
    rows = labels.load_labels()
    by_item = {r["item"]: r for r in rows}
    assert set(by_item) == {"stance", "front_foot_landing", "contact", "head_over_knee_landing",
                            "head_over_knee_contact", "overall"}
    assert by_item["front_foot_landing"]["coach_label"] == "41" and by_item["overall"]["notes"] == "late hands"
    assert by_item["head_over_knee_landing"]["coach_label"] == "pass"
    assert by_item["head_over_knee_landing"]["ai_label"] in ("pass", "fail")

    save(v["state"], rules=["fail"] + [None] * 15, overall=None, confirmed=False)           # relabel
    assert [(r["item"], r["coach_label"]) for r in labels.load_labels()] == [("head_over_knee_landing", "fail")]


def test_reloading_prefills_the_coachs_earlier_answers(loaded, monkeypatch, csv_path):
    video, v = loaded
    save(v["state"], overall=64.0)
    again = tab.load_clip(str(video), "forward_defence", "alex", previous_state=v["state"])
    assert again["confirm"]["value"] is True and again["overall"]["value"] == 64.0
    assert again["rule_0"]["value"] == "pass" and again["rule_1"]["value"] == "fail" and again["rule_2"]["value"] is None
    assert again["stance_slider"]["value"] == 12 and "earlier labels" in again["status"]
    other = tab.load_clip(str(video), "forward_defence", "sam")
    assert other["rule_0"]["value"] is None and other["confirm"]["value"] is False


def test_saving_validates_input(loaded, csv_path):
    _, v = loaded
    assert save(None) == "Load a clip first."
    assert "coach ID" in save(v["state"], coach_id=" ")
    nothing = save(v["state"], confirmed=False, overall=None, rules=["skip"] + [None] * 15)
    assert nothing.startswith("Nothing to save")
    assert not csv_path.exists()


def test_a_locked_file_gives_a_friendly_message(loaded, monkeypatch, csv_path):
    _, v = loaded

    def locked(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(labels, "save_labels", locked)
    assert "open in Excel" in save(v["state"])
