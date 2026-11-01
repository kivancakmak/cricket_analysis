"""Key-moment tests on a synthetic side-on batter with a known stride (no video needed)."""
import numpy as np
import pytest

from src.orientation import Orientation
from src.phases import KeyMoments, detect_key_moments
from src.pose import NUM_LANDMARKS, PoseSequence

SIZE, FPS, N = 1000, 30.0, 100
STRIDE = (30, 50)      # front ankle travels towards the bowler between these frames
HANDS = (36, 52)       # hands drive forward between these frames
NOSE_Y, ANKLE_Y = 0.2, 0.9   # body height = 0.7 of the frame

RIGHT_FACING = Orientation("right", 0.9, "left", 0.9, "right", 0.9)
LEFT_FACING = Orientation("left", 0.9, "right", 0.9, "left", 0.9)


def _ramp(lo, hi):
    """0 before lo, 1 after hi, raised-cosine in between (speed peaks mid-way)."""
    t = np.clip((np.arange(N) - lo) / (hi - lo), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def make_batter(facing="right", stride=True, missing=()):
    d = 1.0 if facing == "right" else -1.0
    front_ankle, back_ankle = (27, 28) if facing == "right" else (28, 27)
    lm = np.zeros((N, NUM_LANDMARKS, 4), dtype=np.float32)
    lm[:, :, 3] = 0.95
    x0 = 0.5
    stride_ramp = _ramp(*STRIDE) if stride else np.zeros(N)

    lm[:, :, 0] = x0
    lm[:, 0, 1] = NOSE_Y
    lm[:, 0, 0] = x0 + d * 0.02
    for idx in (23, 24):                                       # hips drift forward late in the stride
        lm[:, idx, 0] = x0 + d * 0.1 * _ramp(STRIDE[0] + 8, STRIDE[1] + 5)
        lm[:, idx, 1] = 0.55
    lm[:, front_ankle, 0] = x0 + d * 0.30 * stride_ramp
    lm[:, front_ankle, 1] = ANKLE_Y - 0.05 * np.sin(np.pi * np.clip(
        (np.arange(N) - STRIDE[0]) / (STRIDE[1] - STRIDE[0]), 0, 1)) * (1 if stride else 0)
    lm[:, back_ankle, 0] = x0 - d * 0.1
    lm[:, back_ankle, 1] = ANKLE_Y
    for idx in (15, 16):
        lm[:, idx, 0] = x0 + d * 0.2 * _ramp(*HANDS)
        lm[:, idx, 1] = 0.5
    lm[:, :, :2] += np.random.default_rng(0).normal(0, 0.0005, (N, NUM_LANDMARKS, 2)).astype(np.float32)

    detected = np.ones(N, dtype=bool)
    for i in missing:
        lm[i], detected[i] = np.nan, False
    return PoseSequence(FPS, SIZE, SIZE, lm, lm.copy(), detected,
                        (np.arange(N) * 1000 / FPS).astype(np.int64))


@pytest.mark.parametrize(("facing", "orientation"), [("right", RIGHT_FACING), ("left", LEFT_FACING)])
def test_moments_are_found_in_order_for_both_directions(facing, orientation):
    moments = detect_key_moments(make_batter(facing), orientation)
    stance, landing, contact = moments.stance, moments.front_foot_landing, moments.contact

    assert stance is not None and landing is not None and contact is not None
    assert STRIDE[0] - 6 <= stance.frame <= STRIDE[0] + 1          # still before the foot moves
    assert STRIDE[1] - 6 <= landing.frame <= STRIDE[1] + 6         # foot has just stopped
    assert landing.frame <= contact.frame <= HANDS[1] + 12         # hands arrive at/after landing
    assert all(m.confidence > 0.3 for m in (stance, landing, contact))
    assert moments.as_key_frames() == {"STANCE": stance.frame, "FRONT FOOT DOWN": landing.frame,
                                       "CONTACT": contact.frame}


def test_works_at_any_distance_from_the_camera():
    """Shrinking the batter (same motion, half the size) must not move the moments."""
    big = make_batter()
    small = make_batter()
    small.landmarks[:, :, :2] = 0.5 + (small.landmarks[:, :, :2] - 0.5) * 0.5
    a = detect_key_moments(big, RIGHT_FACING)
    b = detect_key_moments(small, RIGHT_FACING)
    assert a.as_key_frames() == b.as_key_frames()


def test_early_hand_movement_does_not_become_the_stance():
    """A bat tap before the stride must not be mistaken for the start of the shot."""
    seq = make_batter()
    seq.landmarks[5:20, 15:17, 0] += 0.05 * np.sin(np.linspace(0, np.pi, 15))[:, None]
    moments = detect_key_moments(seq, RIGHT_FACING)
    assert STRIDE[0] - 6 <= moments.stance.frame <= STRIDE[0] + 1


def test_detection_gaps_are_tolerated():
    moments = detect_key_moments(make_batter(missing=range(10, 15)), RIGHT_FACING)
    assert moments.stance is not None and moments.contact is not None


def test_no_stride_reports_nothing_instead_of_guessing():
    moments = detect_key_moments(make_batter(stride=False), RIGHT_FACING)
    assert moments.stance is None and moments.front_foot_landing is None and moments.contact is None
    assert any("stride" in note for note in moments.notes)


def test_unknown_orientation_gives_no_moments():
    unknown = Orientation("unknown", 0, "unknown", 0, "unknown", 0)
    moments = detect_key_moments(make_batter(), unknown)
    assert isinstance(moments, KeyMoments)
    assert moments.as_key_frames() == {} and moments.signals is None
