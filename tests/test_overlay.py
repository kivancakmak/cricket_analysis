import subprocess

import cv2
import numpy as np
import pytest

from src.overlay import _ffmpeg_exe, draw_skeleton, export_key_frames, render_overlay
from src.pose import NUM_LANDMARKS, PoseSequence

W, H, FPS, N = 320, 240, 20.0, 12


def _sequence(missing=()):
    lm = np.zeros((N, NUM_LANDMARKS, 4), dtype=np.float32)
    lm[:, :, 0] = np.linspace(0.3, 0.7, NUM_LANDMARKS)[None, :]
    lm[:, :, 1] = np.linspace(0.2, 0.9, NUM_LANDMARKS)[None, :]
    lm[:, :, 3] = 0.95
    detected = np.ones(N, dtype=bool)
    for i in missing:
        lm[i] = np.nan
        detected[i] = False
    return PoseSequence(FPS, W, H, lm, lm.copy(), detected,
                        (np.arange(N) * 50).astype(np.int64))


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        writer.write(np.full((H, W, 3), 40 + i * 5, np.uint8))
    writer.release()
    return path


def test_draw_skeleton_marks_pixels_and_handles_none():
    frame = np.zeros((H, W, 3), np.uint8)
    assert draw_skeleton(frame.copy(), None).sum() == 0
    assert draw_skeleton(frame, _sequence().frame(0)).sum() > 0


def test_low_visibility_is_fainter():
    strong, weak = _sequence().frame(0).copy(), _sequence().frame(0).copy()
    weak[:, 3] = 0.1
    bg = np.full((H, W, 3), 20, np.uint8)
    diff_strong = np.abs(draw_skeleton(bg.copy(), strong).astype(int) - 20).sum()
    diff_weak = np.abs(draw_skeleton(bg.copy(), weak).astype(int) - 20).sum()
    assert 0 < diff_weak < diff_strong


def test_render_overlay_is_h264_and_holds_key_frames(video, tmp_path):
    out = tmp_path / "out" / "o.mp4"
    render_overlay(video, _sequence(missing=(3,)), out, key_frames={"STANCE": 2, "CONTACT": 8},
                   annotations={8: ["Head over knee"]})

    # `ffmpeg -i` with no output file exits non-zero by design; we only read its stream info.
    info = subprocess.run([_ffmpeg_exe(), "-i", str(out)], capture_output=True, text=True,
                          check=False).stderr
    assert "h264" in info and "yuv420p" in info

    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    hold = round(0.5 * FPS)
    assert frames == N + 2 * (hold - 1)


def test_export_key_frames(video, tmp_path):
    paths = export_key_frames(video, _sequence(), {"FRONT FOOT DOWN": 4}, tmp_path / "png")
    image = cv2.imread(str(paths["FRONT FOOT DOWN"]))
    assert image.shape[:2] == (H, W)
