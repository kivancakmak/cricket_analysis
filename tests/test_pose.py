import numpy as np

from src.pose import NUM_LANDMARKS, PoseSequence, smooth_pose


def _make_seq(n=60, fps=30.0, noise=0.01, seed=0, missing=()):
    """Synthetic sequence: every landmark drifts linearly in x, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    clean = np.zeros((n, NUM_LANDMARKS, 4), dtype=np.float32)
    clean[:, :, 0] = (0.2 + 0.5 * t)[:, None]
    clean[:, :, 1] = 0.5
    clean[:, :, 3] = 0.9
    noisy = clean.copy()
    noisy[:, :, :3] += rng.normal(0, noise, size=(n, NUM_LANDMARKS, 3)).astype(np.float32)
    detected = np.ones(n, dtype=bool)
    for i in missing:
        detected[i] = False
        noisy[i] = np.nan
    seq = PoseSequence(
        fps=fps, width=640, height=480, landmarks=noisy, world_landmarks=noisy.copy(),
        detected=detected, timestamps_ms=(np.arange(n) * 1000 / fps).astype(np.int64),
    )
    return seq, clean


def test_smoothing_reduces_jitter_and_keeps_raw():
    seq, clean = _make_seq()
    raw_before = seq.landmarks.copy()
    smoothed = smooth_pose(seq)

    np.testing.assert_array_equal(seq.landmarks, raw_before)  # input untouched
    err_raw = np.abs(seq.landmarks[:, :, :3] - clean[:, :, :3]).mean()
    err_smooth = np.abs(smoothed.landmarks[:, :, :3] - clean[:, :, :3]).mean()
    assert err_smooth < err_raw


def test_smoothing_preserves_visibility_and_gaps():
    seq, _ = _make_seq(missing=(20, 21, 22))
    smoothed = smooth_pose(seq)

    assert not smoothed.detected[20:23].any()
    assert np.isnan(smoothed.landmarks[20:23]).all()
    assert smoothed.detected[:20].all() and smoothed.detected[23:].all()
    np.testing.assert_array_equal(smoothed.landmarks[:, :, 3][seq.detected],
                                  seq.landmarks[:, :, 3][seq.detected])


def test_smoothing_follows_fast_motion():
    """A step change should be tracked, not smeared (beta lets cutoff rise)."""
    seq, _ = _make_seq(noise=0.0)
    seq.landmarks[30:, :, 0] += 0.3
    smoothed = smooth_pose(seq)
    assert abs(smoothed.landmarks[40, 0, 0] - seq.landmarks[40, 0, 0]) < 0.05


def test_save_load_roundtrip(tmp_path):
    seq, _ = _make_seq(missing=(5,))
    path = tmp_path / "seq.npz"
    seq.save(path)
    loaded = PoseSequence.load(path)

    assert (loaded.fps, loaded.width, loaded.height) == (seq.fps, seq.width, seq.height)
    np.testing.assert_array_equal(loaded.detected, seq.detected)
    np.testing.assert_array_equal(loaded.landmarks, seq.landmarks)
    assert loaded.frame(5) is None and loaded.frame(6) is not None
    assert 0.0 < loaded.detection_rate() < 1.0
