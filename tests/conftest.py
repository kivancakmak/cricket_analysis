"""Shared fixtures: a synthetic batter that passes the quality gate, and blank test videos."""
import cv2
import numpy as np
import pytest

from src.pose import PoseSequence
from tests.test_orientation import make_batter


def good_sequence() -> PoseSequence:
    """A fully visible right-hander facing right: stance, stride at frames 30-50, hands drive forward."""
    seq = make_batter("right", "right")
    lm = seq.landmarks
    unset = (lm[:, :, 0] == 0) & (lm[:, :, 1] == 0)          # landmarks the helper did not place
    lm[unset, 0] = 0.5
    lm[unset, 1] = 0.5
    ramp = np.clip((np.arange(seq.n_frames) - 36) / 16, 0, 1)
    for wrist in (15, 16):
        lm[:, wrist, 0] = 0.5 + 0.1 * (0.5 - 0.5 * np.cos(np.pi * ramp))
        lm[:, wrist, 1] = 0.5
        lm[:, wrist, 3] = 0.95
    for knee, x in ((25, 0.52), (26, 0.45)):
        lm[:, knee, 0], lm[:, knee, 1] = x, 0.72
    seq.world_landmarks[:] = lm
    return seq


def write_blank_video(path, frames: int, fps: float = 30.0, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), 40 + i % 50, np.uint8))
    writer.release()
    return path


@pytest.fixture
def make_sequence():
    return good_sequence


@pytest.fixture
def blank_video(tmp_path):
    def make(frames=90, fps=30.0, name="clip.mp4"):
        return write_blank_video(tmp_path / name, frames, fps)
    return make
