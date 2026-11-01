"""Metric tests on hand-built poses where the right answer is known.

Poses use a 1000x1000 frame and normalized coordinates. The default batter faces
image-right with the left leg in front; a 0.25-tall torso (shoulders y=0.30, hips
y=0.55) is therefore 250 px, and nose-to-ankle is 0.7 of the frame (700 px).
"""
import math

import numpy as np
import pytest

from src.metrics import compute_metrics
from src.orientation import Orientation
from src.phases import KeyMoment, KeyMoments
from src.pose import NUM_LANDMARKS, PoseSequence

N, FPS, SIZE = 12, 30.0, 1000
LANDING, CONTACT = 5, 9

# landmark index -> (x, y); everything else sits at (0.5, 0.5)
BASE = {
    0: (0.66, 0.20),                       # nose
    7: (0.62, 0.20), 8: (0.62, 0.20),      # ears
    11: (0.50, 0.30), 12: (0.50, 0.30),    # shoulders
    13: (0.55, 0.40), 14: (0.55, 0.40),    # elbows
    15: (0.60, 0.50), 16: (0.60, 0.50),    # wrists
    23: (0.50, 0.55), 24: (0.50, 0.55),    # hips
    25: (0.60, 0.72), 26: (0.45, 0.72),    # knees  (left = front)
    27: (0.75, 0.90), 28: (0.35, 0.90),    # ankles (left = front)
}

RIGHT_FACING_RIGHTY = Orientation("right", 0.9, "left", 0.9, "right", 0.9)
LEFT_FACING_RIGHTY = Orientation("left", 0.9, "left", 0.9, "right", 0.9)
RIGHT_FACING_LEFTY = Orientation("right", 0.9, "right", 0.9, "left", 0.9)


def build(overrides=None, vis=None, mirror=False, swap=False, scale=1.0, nose_shift_after=None,
          nose_drop_from=None):
    pts = {**BASE, **(overrides or {})}
    if swap:   # left <-> right landmark labels (left-handed batter)
        pairs = {7: 8, 8: 7, **{i: i + 1 if i % 2 else i - 1 for i in range(11, 33)}}
        pts = {pairs.get(i, i): xy for i, xy in pts.items()}
    lm = np.full((N, NUM_LANDMARKS, 4), 0.5, dtype=np.float64)
    lm[:, :, 3] = 0.95
    for i, (x, y) in pts.items():
        lm[:, i, 0], lm[:, i, 1] = (1 - x if mirror else x), y
    lm[:, :, :2] = 0.5 + (lm[:, :, :2] - 0.5) * scale
    for i, v in (vis or {}).items():
        lm[:, i, 3] = v
    if nose_drop_from is not None:   # head sinks into a lunge for most of the clip
        lm[nose_drop_from:, 0, 1] += 0.25
    if nose_shift_after is not None:
        lm[nose_shift_after:, 0, 0] += (-1 if mirror else 1) * 0.025
    return PoseSequence(FPS, SIZE, SIZE, lm.astype(np.float32), lm.astype(np.float32),
                        np.ones(N, dtype=bool), (np.arange(N) * 1000 / FPS).astype(np.int64))


def moments(landing_conf=1.0, with_contact=True):
    return KeyMoments(
        KeyMoment(2, 2 / FPS, 1.0),
        KeyMoment(LANDING, LANDING / FPS, landing_conf),
        KeyMoment(CONTACT, CONTACT / FPS, 1.0) if with_contact else None,
    )


def measure(overrides=None, orientation=RIGHT_FACING_RIGHTY, **build_kwargs):
    return compute_metrics(build(overrides, **build_kwargs), orientation, moments())


def test_head_over_front_knee_uses_ears_and_is_signed():
    m = measure({25: (0.50, 0.72)})["head_over_front_knee_landing"]
    assert m.value == pytest.approx((0.62 - 0.50) / 0.25)            # head ahead of the knee
    behind = measure({25: (0.70, 0.72)})["head_over_front_knee_landing"]
    assert behind.value == pytest.approx((0.62 - 0.70) / 0.25)       # head behind the knee


def test_head_falls_back_to_nose_when_ears_are_hidden():
    m = measure({25: (0.50, 0.72)}, vis={7: 0.0, 8: 0.0})["head_over_front_knee_landing"]
    assert m.value == pytest.approx((0.66 - 0.50) / 0.25)


@pytest.mark.parametrize(("ankle", "expected"), [((0.70, 0.75), 90.0), ((0.50, 0.95), 180.0)])
def test_front_knee_angle(ankle, expected):
    m = measure({23: (0.5, 0.55), 25: (0.5, 0.75), 27: ankle})["front_knee_angle_landing"]
    assert m.value == pytest.approx(expected, abs=1e-3)


def test_stride_length_is_ankle_gap_over_body_height():
    assert measure()["stride_length_landing"].value == pytest.approx(0.40 / 0.70)


@pytest.mark.parametrize(("hip_x", "expected"), [(0.35, 0.0), (0.55, 0.5), (0.75, 1.0)])
def test_balance_between_the_feet(hip_x, expected):
    m = measure({23: (hip_x, 0.55), 24: (hip_x, 0.55)})
    assert m["balance_landing"].value == pytest.approx(expected)
    assert m["balance_contact"].value == pytest.approx(expected)


def test_balance_unavailable_when_feet_are_together():
    result = measure({27: (0.5, 0.9), 28: (0.5, 0.9)})
    assert result.get("balance_landing") is None
    assert any("balance_landing" in note for note in result.notes)


@pytest.mark.parametrize(("elbow_y", "expected"), [(0.40, 0.5), (0.30, 1.0), (0.25, 1.25)])
def test_front_elbow_height(elbow_y, expected):
    m = measure({13: (0.55, elbow_y)})["front_elbow_height_contact"]
    assert m.value == pytest.approx(expected)


def test_spine_tilt_leans_towards_the_bowler():
    lean = math.radians(20)
    shoulders = (0.5 + 0.25 * math.sin(lean), 0.55 - 0.25 * math.cos(lean))
    forward = measure({11: shoulders, 12: shoulders})["spine_tilt_landing"].value
    assert forward == pytest.approx(20.0, abs=1e-3)
    assert measure()["spine_tilt_landing"].value == pytest.approx(0.0, abs=1e-6)
    backward = measure({11: (1 - shoulders[0], shoulders[1]), 12: (1 - shoulders[0], shoulders[1])})
    assert backward["spine_tilt_landing"].value == pytest.approx(-20.0, abs=1e-3)


def test_head_stability_is_largest_movement_since_landing():
    result = compute_metrics(build(nose_shift_after=7), RIGHT_FACING_RIGHTY, moments())
    assert result["head_stability"].value == pytest.approx(0.025 * SIZE / 250)
    assert measure()["head_stability"].value == pytest.approx(0.0, abs=1e-6)


def test_confidence_is_weakest_joint_times_moment_confidence():
    result = compute_metrics(build(vis={25: 0.4}), RIGHT_FACING_RIGHTY, moments(landing_conf=0.8))
    # weakest used joint is the front knee (0.4); moment factor = 0.5 + 0.5 * 0.8
    assert result["head_over_front_knee_landing"].confidence == pytest.approx(0.4 * 0.9)
    # a metric that does not use the knee is unaffected
    assert result["stride_length_landing"].confidence == pytest.approx(0.95 * 0.9)


def test_body_scale_is_the_upright_size_not_the_lunge_size():
    """The head sinks for most of the clip; stride must still be relative to upright height."""
    result = compute_metrics(build(nose_drop_from=3), RIGHT_FACING_RIGHTY, moments())
    assert result["stride_length_landing"].value == pytest.approx(0.40 / 0.70)


def test_metrics_do_not_depend_on_camera_distance():
    near, far = measure(), measure(scale=0.5)
    assert near.values.keys() == far.values.keys()
    for name in near.values:
        assert far[name].value == pytest.approx(near[name].value, abs=1e-4), name


@pytest.mark.parametrize(("kwargs", "orientation"), [
    ({"mirror": True}, LEFT_FACING_RIGHTY),                     # batter facing image-left
    ({"swap": True}, RIGHT_FACING_LEFTY),                       # left-handed batter
    ({"mirror": True, "swap": True}, Orientation("left", 0.9, "right", 0.9, "left", 0.9)),
])
def test_same_technique_gives_same_numbers_in_every_orientation(kwargs, orientation):
    reference = measure()
    other = measure(orientation=orientation, **kwargs)
    assert other.values.keys() == reference.values.keys()
    for name in reference.values:
        assert other[name].value == pytest.approx(reference[name].value, abs=1e-4), name


def test_untracked_joint_drops_only_the_metrics_that_need_it():
    seq = build()
    seq.landmarks[:, 25] = np.nan                       # front knee lost
    result = compute_metrics(seq, RIGHT_FACING_RIGHTY, moments())
    assert result.get("front_knee_angle_landing") is None
    assert result.get("head_over_front_knee_contact") is None
    assert result.get("stride_length_landing") is not None
    assert any("front_knee_angle_landing" in note for note in result.notes)


def test_missing_contact_skips_contact_metrics():
    result = compute_metrics(build(), RIGHT_FACING_RIGHTY, moments(with_contact=False))
    assert result.get("balance_contact") is None and result.get("head_stability") is None
    assert result.get("balance_landing") is not None


def test_unknown_orientation_gives_no_metrics():
    unknown = Orientation("unknown", 0, "unknown", 0, "unknown", 0)
    result = compute_metrics(build(), unknown, moments())
    assert result.values == {} and result.notes
