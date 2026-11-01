"""Quality-gate tests on synthetic sequences: one good clip, then one defect at a time."""
import numpy as np
import pytest

from src.metrics import Metric, Metrics
from src.orientation import Orientation
from src.phases import KeyMoment, KeyMoments
from src.pose import NUM_LANDMARKS, PoseSequence
from src.quality import assess_quality

FPS, SIZE = 30.0, 1000
GOOD_ORIENTATION = Orientation("right", 0.9, "left", 0.9, "right", 0.9)


def make_seq(seconds=4.0, vis=0.95, detected_fraction=1.0, shoulder_dx=0.03):
    """A static, fully visible batter well inside the frame (shoulders 2*shoulder_dx apart)."""
    n = round(seconds * FPS)
    lm = np.full((n, NUM_LANDMARKS, 4), 0.5, dtype=np.float32)
    lm[:, :, 3] = vis
    pts = {0: (0.55, 0.20), 7: (0.52, 0.20), 8: (0.52, 0.20),
           11: (0.5 - shoulder_dx, 0.30), 12: (0.5 + shoulder_dx, 0.30),
           15: (0.58, 0.50), 16: (0.58, 0.50), 23: (0.5, 0.55), 24: (0.5, 0.55),
           25: (0.6, 0.72), 26: (0.45, 0.72), 27: (0.75, 0.88), 28: (0.35, 0.88),
           29: (0.74, 0.90), 30: (0.34, 0.90), 31: (0.78, 0.90), 32: (0.38, 0.90)}
    for i, (x, y) in pts.items():
        lm[:, i, 0], lm[:, i, 1] = x, y
    detected = np.arange(n) < round(n * detected_fraction)
    lm[~detected] = np.nan
    return PoseSequence(FPS, SIZE, SIZE, lm, lm.copy(), detected, (np.arange(n) * 1000 / FPS).astype(np.int64))


def good_moments(conf=0.8):
    return KeyMoments(KeyMoment(10, 0.3, conf), KeyMoment(50, 1.7, conf), KeyMoment(60, 2.0, conf))


def good_metrics(stride=0.8):
    return Metrics({"stride_length_landing": Metric(stride, 0.9)})


def assess(seq=None, orientation=GOOD_ORIENTATION, moments="default", metrics="default", config=None):
    return assess_quality(
        seq or make_seq(), orientation,
        good_moments() if moments == "default" else moments,
        good_metrics() if metrics == "default" else metrics, config)


def names(report):
    return [c.name for c in report.problems]


def test_good_clip_passes_with_no_message():
    report = assess()
    assert report.ok and report.message == "" and report.instructions == []
    assert [c.name for c in report.checks] == ["duration", "detection", "frame", "visibility",
                                               "camera", "key_moments"]


@pytest.mark.parametrize(("seconds", "word"), [(1.0, "too short"), (12.0, "too long")])
def test_clip_length(seconds, word):
    report = assess(make_seq(seconds=seconds))
    assert names(report) == ["duration"]
    assert word in report.message and report.message.startswith("Please film again.")


def test_clip_length_boundaries_are_inclusive():
    assert assess(make_seq(seconds=2.0)).ok and assess(make_seq(seconds=8.0)).ok


def test_too_few_frames_with_a_person():
    report = assess(make_seq(detected_fraction=0.6))
    assert names(report) == ["detection"]
    assert "60%" in report.message
    assert assess(make_seq(detected_fraction=0.85)).ok


def test_hidden_feet_are_named_in_the_message():
    seq = make_seq()
    seq.landmarks[:, [27, 28], 3] = 0.2         # both ankles barely visible
    report = assess(seq)
    assert names(report) == ["visibility"]
    assert "your feet" in report.message


def test_near_side_joint_is_enough_when_the_far_side_is_hidden():
    seq = make_seq()
    seq.landmarks[:, [28, 26, 24, 16], 3] = 0.05     # the whole far side is occluded
    assert assess(seq).ok


@pytest.mark.parametrize(("joint", "y", "x", "expected"), [
    (27, 0.999, 0.75, "Move the camera back so your feet are visible"),    # feet cut off at the bottom
    (0, 0.002, 0.55, "head stays in the frame"),                           # head cut off at the top
    (27, 0.88, 0.002, "inside the frame"),                                 # walked out on the left
])
def test_batter_must_stay_inside_the_frame(joint, y, x, expected):
    seq = make_seq()
    seq.landmarks[:, joint, 0], seq.landmarks[:, joint, 1] = x, y
    report = assess(seq)
    assert names(report) == ["frame"]
    assert expected in report.message


def test_batter_leaving_the_frame_reports_the_framing_not_the_detection_rate():
    seq = make_seq(detected_fraction=0.6)
    seq.landmarks[seq.detected, 27, 0] = 0.002          # the frames we do see are at the left edge
    report = assess(seq)
    assert {"detection", "frame"} <= set(names(report))
    assert [c.name for c in report.root_problems] == ["frame"]


def test_brief_touch_of_the_border_is_tolerated():
    seq = make_seq()
    seq.landmarks[:5, 27, 1] = 0.999            # 5 of 120 frames
    assert assess(seq).ok


def test_chest_on_camera_is_rejected():
    report = assess(make_seq(shoulder_dx=0.22))          # shoulders 0.44 wide vs torso 0.25 -> ratio 1.76
    assert names(report) == ["camera"] and "side-on" in report.message


def test_camera_looking_down_the_pitch_is_rejected_by_the_foot_gap():
    report = assess(metrics=good_metrics(stride=0.1))    # feet barely apart sideways
    assert names(report) == ["camera"]
    assert "0.10 body heights" in report.problems[0].detail


def test_camera_check_still_works_without_metrics():
    assert assess(metrics=None).ok


@pytest.mark.parametrize(("missing", "advice"), [
    ("stance", "settle before the shot"),
    ("front_foot_landing", "front foot lands"),
    ("contact", "moment of the shot"),
])
def test_missing_or_unsure_key_moments(missing, advice):
    moments = {"stance": good_moments().stance, "front_foot_landing": good_moments().front_foot_landing,
               "contact": good_moments().contact}
    moments[missing] = None
    report = assess(moments=KeyMoments(**moments))
    assert names(report) == ["key_moments"] and advice in report.message

    moments[missing] = KeyMoment(30, 1.0, 0.1)           # found, but not trusted
    assert names(assess(moments=KeyMoments(**moments))) == ["key_moments"]


def test_unknown_orientation_asks_for_a_side_view():
    unknown = Orientation("unknown", 0.0, "unknown", 0.0, "unknown", 0.0)
    report = assess(orientation=unknown, moments=KeyMoments(None, None, None))
    assert names(report) == ["key_moments"] and "which way you're facing" in report.message


def test_only_root_causes_are_reported():
    """Cut-off feet also give low ankle visibility and unreliable moments: show one instruction."""
    seq = make_seq()
    seq.landmarks[:, [27, 28], 1] = 0.999
    seq.landmarks[:, [27, 28], 3] = 0.1
    report = assess(seq, moments=KeyMoments(None, None, None))
    assert set(names(report)) == {"frame", "visibility", "key_moments"}      # all recorded ...
    assert [c.name for c in report.root_problems] == ["frame"]               # ... one shown
    assert len(report.instructions) == 1 and "feet are visible" in report.message


def test_independent_problems_are_all_reported_most_important_first():
    report = assess(make_seq(seconds=12.0), moments=KeyMoments(None, None, None))
    assert [c.name for c in report.root_problems] == ["duration"]       # moments are a consequence
    seq = make_seq(seconds=12.0)
    seq.landmarks[:, 0, 1] = 0.002                                      # head also cut off
    report = assess(seq)
    assert [c.name for c in report.root_problems] == ["duration", "frame"]
    assert report.message.startswith("Please film again. The clip is too long")
    assert len(report.instructions) == 2


def test_no_person_at_all_is_one_clear_message_not_a_crash():
    seq = make_seq(detected_fraction=0.0)
    report = assess(seq, orientation=Orientation("unknown", 0, "unknown", 0, "unknown", 0),
                    moments=KeyMoments(None, None, None), metrics=None)
    assert [c.name for c in report.root_problems] == ["detection"]


def test_thresholds_come_from_the_config():
    strict = {"min_duration_s": 5.0, "max_duration_s": 8.0, "min_detection_rate": 0.8,
              "min_joint_visibility": 0.5, "border_margin": 0.01, "max_border_fraction": 0.1,
              "max_shoulder_width_ratio": 0.9, "min_stride_projection": 0.3,
              "min_orientation_confidence": 0.3, "min_moment_confidence": 0.25}
    assert names(assess(config=strict)) == ["duration"]
