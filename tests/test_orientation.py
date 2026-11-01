"""Orientation tests on synthetic side-on batters (no video or model needed)."""
import numpy as np
import pytest

from src.orientation import detect_orientation
from src.pose import NUM_LANDMARKS, PoseSequence

W, H, FPS, N = 1920, 1080, 30.0, 90
LEFT_SIDE = {"shoulder": 11, "hip": 23, "ankle": 27, "heel": 29, "toe": 31, "ear": 7}
RIGHT_SIDE = {"shoulder": 12, "hip": 24, "ankle": 28, "heel": 30, "toe": 32, "ear": 8}


def make_batter(facing="right", handed="right", stride=True, far_visibility=0.95,
                missing=(), seed=0):
    """Synthetic side-on batter.

    The front shoulder/hip/ankle (the batter's left side for a right-hander) sit further
    towards the bowler (``facing`` direction) than the back ones; the head is in
    profile; during the shot the front ankle strides towards the bowler.
    """
    rng = np.random.default_rng(seed)
    d = 1.0 if facing == "right" else -1.0
    front, back = (LEFT_SIDE, RIGHT_SIDE) if handed == "right" else (RIGHT_SIDE, LEFT_SIDE)
    torso = 0.25 * H / W        # normalized-x length of a 0.25-high torso (isotropic pixels)
    cx = 0.5

    lm = np.zeros((N, NUM_LANDMARKS, 4), dtype=np.float32)
    lm[:, :, 3] = 0.95
    t = np.clip((np.arange(N) - 30) / 20, 0, 1) if stride else np.zeros(N)   # stride 30..50

    def put(idx, x, y):
        lm[:, idx, 0], lm[:, idx, 1] = x, y

    put(0, cx + d * 0.2 * torso, 0.20)                 # nose, ahead of the ears
    for side in (front, back):
        put(side["ear"], cx, 0.20)
    put(front["shoulder"], cx + d * 0.15 * torso, 0.30)
    put(back["shoulder"], cx - d * 0.15 * torso, 0.30)
    put(front["hip"], cx + d * 0.1 * torso, 0.55)
    put(back["hip"], cx - d * 0.1 * torso, 0.55)
    put(front["ankle"], cx + d * 0.5 * torso + d * 1.2 * torso * t, 0.90)
    put(back["ankle"], cx - d * 0.5 * torso + d * 0.2 * torso * t, 0.90)
    for side in (front, back):
        put(side["heel"], lm[:, side["ankle"], 0], 0.92)
        put(side["toe"], lm[:, side["ankle"], 0] + d * 0.05 * torso, 0.92)
    # the far-side limbs are partly hidden from the camera
    for name in ("shoulder", "hip", "ankle", "heel", "toe", "ear"):
        lm[:, back[name], 3] = far_visibility

    lm[:, :, :2] += rng.normal(0, 0.002, size=(N, NUM_LANDMARKS, 2)).astype(np.float32)
    detected = np.ones(N, dtype=bool)
    for i in missing:
        lm[i], detected[i] = np.nan, False
    return PoseSequence(FPS, W, H, lm, lm.copy(), detected, (np.arange(N) * 1000 / FPS).astype(np.int64))


@pytest.mark.parametrize("facing", ["left", "right"])
@pytest.mark.parametrize("handed", ["right", "left"])
def test_all_orientations(facing, handed):
    result = detect_orientation(make_batter(facing, handed))
    assert result.facing == facing
    assert result.handedness == handed
    assert result.front_leg == ("left" if handed == "right" else "right")
    assert result.back_leg == ("right" if handed == "right" else "left")
    assert result.confidence > 0.5


@pytest.mark.parametrize("facing", ["left", "right"])
def test_stationary_stance_still_works(facing):
    """No stride at all: the stance cues alone must give the answer."""
    result = detect_orientation(make_batter(facing, "right", stride=False))
    assert (result.facing, result.front_leg, result.handedness) == (facing, "left", "right")


def test_occluded_far_side_lowers_confidence():
    clear = detect_orientation(make_batter("left", "right", far_visibility=0.95))
    hidden = detect_orientation(make_batter("left", "right", far_visibility=0.4))
    assert hidden.facing == "left" and hidden.front_leg == "left"
    assert hidden.confidence < clear.confidence


def test_gaps_in_detection_are_tolerated():
    result = detect_orientation(make_batter("right", "left", missing=range(5, 20)))
    assert (result.facing, result.front_leg, result.handedness) == ("right", "right", "left")


def test_no_detections_is_unknown():
    seq = make_batter()
    seq.detected[:] = False
    seq.landmarks[:] = np.nan
    result = detect_orientation(seq)
    assert (result.facing, result.front_leg, result.handedness) == ("unknown",) * 3
    assert result.confidence == 0.0


def test_ambiguous_pose_is_unknown_not_a_guess():
    """A perfectly symmetric, motionless pose carries no direction information."""
    seq = make_batter(stride=False)
    seq.landmarks[:, :, 0] = 0.5
    seq.landmarks[:, :, 1] = np.linspace(0.1, 0.9, NUM_LANDMARKS)
    result = detect_orientation(seq)
    assert result.facing == "unknown"
