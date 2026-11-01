"""Stage 5: key moments of the shot from body-keypoint motion.

``detect_key_moments`` finds three frames, each with a heuristic confidence:

* STANCE               - the last stable frame before the front foot starts its stride.
* FRONT_FOOT_LANDING   - the front foot has stopped moving forward and is on the ground.
* CONTACT              - approximate bat-ball contact (no bat or ball is tracked).

How it works
------------
All motion is measured in *body heights per second* (positions divided by the
batter's body height, so it works at any distance from the camera) and
"forward" means towards the bowler, taken from the orientation result.

1. The front ankle's forward speed is computed from lightly smoothed positions.
   Its maximum is the *stride peak*: the foot swinging towards the bowler.
2. STANCE: walking back from the stride peak, the last run of
   ``stance_hold_frames`` frames in which both the front ankle and the hips are
   nearly still. Anything before that (bat tap, waggle, small shuffles) is ignored
   because the search is anchored on the stride, not on the first movement.
3. FRONT_FOOT_LANDING: walking forward from the stride peak, the first run of
   ``landing_hold_frames`` frames where the ankle's forward speed has dropped to a
   small fraction of the peak *and* its vertical speed is near zero.
4. CONTACT: the hands (mean of both wrists) drive forward, then stop as the bat
   meets the ball. The fastest forward hand movement is found in a window around
   landing; contact is the first frame at or after both that peak and the
   landing where hand speed has fallen to ``contact_release_fraction`` of the
   peak (the hands have "arrived"). On the sample clips the fastest hand movement
   comes 0.25-0.35 s *before* the foot lands, so the peak alone would always sit
   at the start of a post-landing window; the release point is what lands at or
   after the foot. It stays approximate until bat and ball are tracked.

Every threshold lives in ``PHASE_CONFIG`` because footage from different cameras
and coaches will need tuning. A confidence is attached to each moment: it
combines landmark visibility, how clear the underlying motion peak is, and
whether a fallback had to be used. Anything that went wrong is listed in
``KeyMoments.notes``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.orientation import L_ANKLE, L_HIP, NOSE, R_ANKLE, R_HIP, Orientation
from src.pose import PoseSequence

L_WRIST, R_WRIST = 15, 16

# Speeds are in body heights per second; durations in seconds or frames as named.
PHASE_CONFIG: dict = {
    "min_visibility": 0.3,         # landmarks below this are treated as missing on that frame
    "smooth_frames": 5,            # moving-average width applied to positions before differentiating
    "min_stride_speed": 0.25,      # slower peak than this -> no stride found
    # STANCE
    "stance_ankle_speed": 0.10,    # front ankle is "still" below this
    "stance_hip_speed": 0.15,      # hips are "still" below this
    "stance_hold_frames": 3,       # consecutive still frames that make a stable stance
    # FRONT_FOOT_LANDING
    "landing_fwd_abs": 0.10,       # forward speed must fall below max(this, rel * peak) ...
    "landing_fwd_rel": 0.15,
    "landing_vert_speed": 0.12,    # ... and vertical speed below this
    "landing_hold_frames": 2,
    # CONTACT
    "contact_pre_s": 0.5,          # look for the hand-speed peak from this long before landing ...
    "contact_window_s": 0.6,       # ... until this long after it
    "contact_release_fraction": 0.15,  # contact = hands slowed to this fraction of their peak
    "min_wrist_speed": 0.2,        # slower peak than this -> no contact found
}


@dataclass(frozen=True)
class KeyMoment:
    """One detected key moment: frame index, time in seconds and confidence (0-1)."""
    frame: int
    time_s: float
    confidence: float


@dataclass(frozen=True)
class PhaseSignals:
    """Motion curves used for detection (body heights/s); kept for debugging and plots."""

    time_s: np.ndarray
    ankle_forward: np.ndarray    # front ankle speed towards the bowler
    ankle_vertical: np.ndarray   # front ankle vertical speed (image y, down is positive)
    ankle_speed: np.ndarray      # front ankle speed, any direction
    hip_speed: np.ndarray        # mid-hip speed, any direction
    wrist_forward: np.ndarray    # mean-of-wrists speed towards the bowler
    body_height_px: float


@dataclass(frozen=True)
class KeyMoments:
    """The three key moments of a shot (None where not found) plus the signals and notes behind them."""
    stance: KeyMoment | None
    front_foot_landing: KeyMoment | None
    contact: KeyMoment | None
    signals: PhaseSignals | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_key_frames(self) -> dict[str, int]:
        """Label -> frame index, in the form ``render_overlay`` expects."""
        labelled = {"STANCE": self.stance, "FRONT FOOT DOWN": self.front_foot_landing,
                    "CONTACT": self.contact}
        return {label: m.frame for label, m in labelled.items() if m is not None}


def _track(seq: PoseSequence, idx: int, min_vis: float) -> np.ndarray:
    """Pixel (x, y) of a landmark per frame; missing frames are interpolated linearly."""
    lm = seq.landmarks[:, idx]
    pts = lm[:, :2] * (seq.width, seq.height)
    ok = seq.detected & (lm[:, 3] >= min_vis) & np.isfinite(pts).all(axis=1)
    out = np.full_like(pts, np.nan, dtype=np.float64)
    if ok.sum() >= 2:
        frames = np.arange(len(pts))
        for c in range(2):
            out[:, c] = np.interp(frames, frames[ok], pts[ok, c])
    return out


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Moving average along axis 0, edges padded by repeating the end values."""
    if width <= 1:
        return x
    pad = width // 2
    padded = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    kernel = np.ones(width) / width
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid")
                     for c in range(x.shape[1])], axis=1) if x.ndim > 1 \
        else np.convolve(padded, kernel, mode="valid")


def _velocity(track: np.ndarray, fps: float, body_height: float, width: int) -> np.ndarray:
    """Smoothed velocity in body heights per second, shape (n, 2)."""
    return np.gradient(_smooth(track / body_height, width), axis=0) * fps


def compute_signals(seq: PoseSequence, orientation: Orientation,
                    config: dict | None = None) -> PhaseSignals | None:
    """Build the motion curves; None if orientation or tracking is unusable."""
    cfg = PHASE_CONFIG if config is None else config
    if orientation.facing == "unknown" or orientation.front_leg == "unknown" or seq.n_frames < 3:
        return None
    mv, width = cfg["min_visibility"], cfg["smooth_frames"]
    d = 1.0 if orientation.facing == "right" else -1.0
    front_ankle = L_ANKLE if orientation.front_leg == "left" else R_ANKLE

    ankle = _track(seq, front_ankle, mv)
    hips = (_track(seq, L_HIP, mv) + _track(seq, R_HIP, mv)) / 2
    nose = _track(seq, NOSE, mv)
    both_ankles = (_track(seq, L_ANKLE, mv) + _track(seq, R_ANKLE, mv)) / 2
    wrists = np.nanmean([_track(seq, L_WRIST, mv), _track(seq, R_WRIST, mv)], axis=0)
    if not all(np.isfinite(a).all() for a in (ankle, hips, wrists)):
        return None

    # Body size: head-to-feet height, median over the clip (stable against crouching).
    heights = both_ankles[:, 1] - nose[:, 1]
    body_height = float(np.nanmedian(heights)) if np.isfinite(heights).any() else np.nan
    if not np.isfinite(body_height) or body_height <= 0:
        return None

    v_ankle = _velocity(ankle, seq.fps, body_height, width)
    v_hips = _velocity(hips, seq.fps, body_height, width)
    v_wrist = _velocity(wrists, seq.fps, body_height, width)
    return PhaseSignals(
        time_s=np.arange(seq.n_frames) / seq.fps,
        ankle_forward=d * v_ankle[:, 0],
        ankle_vertical=v_ankle[:, 1],
        ankle_speed=np.hypot(v_ankle[:, 0], v_ankle[:, 1]),
        hip_speed=np.hypot(v_hips[:, 0], v_hips[:, 1]),
        wrist_forward=d * v_wrist[:, 0],
        body_height_px=body_height,
    )


def _first_run(mask: np.ndarray, start: int, length: int) -> int | None:
    """First index >= start beginning a run of ``length`` True values, else None."""
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run == length:
            return i - length + 1
    return None


def _last_run_end(mask: np.ndarray, stop: int, length: int) -> int | None:
    """Last index < stop ending a run of ``length`` True values, else None."""
    run = 0
    for i in range(stop - 1, -1, -1):
        run = run + 1 if mask[i] else 0
        if run == length:
            return i + length - 1
    return None


def _mean_visibility(seq: PoseSequence, indices: tuple[int, ...]) -> float:
    vis = seq.landmarks[seq.detected][:, indices, 3]
    return float(np.clip(vis.mean(), 0, 1)) if vis.size else 0.0


def detect_key_moments(seq: PoseSequence, orientation: Orientation,
                       config: dict | None = None) -> KeyMoments:
    """Find STANCE, FRONT_FOOT_LANDING and CONTACT frames (see the module docstring)."""
    cfg = PHASE_CONFIG if config is None else config
    sig = compute_signals(seq, orientation, cfg)
    if sig is None:
        return KeyMoments(None, None, None, None,
                          ("Orientation or body tracking unusable; no key moments found.",))

    notes: list[str] = []
    fps = seq.fps
    n = seq.n_frames

    def moment(frame: int, confidence: float) -> KeyMoment:
        """Build a ``KeyMoment`` with the confidence clipped to 0-1."""
        return KeyMoment(int(frame), float(frame / fps), float(np.clip(confidence, 0.0, 1.0)))

    front_ankle = L_ANKLE if orientation.front_leg == "left" else R_ANKLE
    ankle_vis = _mean_visibility(seq, (front_ankle,))
    wrist_vis = _mean_visibility(seq, (L_WRIST, R_WRIST))
    hip_vis = _mean_visibility(seq, (L_HIP, R_HIP))

    # --- stride peak: the anchor for stance and landing -------------------------------
    peak = int(np.argmax(sig.ankle_forward))
    peak_speed = float(sig.ankle_forward[peak])
    if peak_speed < cfg["min_stride_speed"]:
        notes.append("No forward stride of the front foot found.")
        return KeyMoments(None, None, None, sig, tuple(notes))
    # How clearly the stride stands out: 1.0 once it is 4x the minimum speed.
    stride_clarity = min(1.0, peak_speed / (4 * cfg["min_stride_speed"]))

    # --- STANCE -----------------------------------------------------------------------
    still = (sig.ankle_speed < cfg["stance_ankle_speed"]) & (sig.hip_speed < cfg["stance_hip_speed"])
    end = _last_run_end(still, peak, cfg["stance_hold_frames"])
    if end is not None:
        recent = slice(end - cfg["stance_hold_frames"] + 1, end + 1)
        margin = 1 - 0.5 * (sig.ankle_speed[recent].mean() / cfg["stance_ankle_speed"]
                            + sig.hip_speed[recent].mean() / cfg["stance_hip_speed"])
        stance = moment(end, stride_clarity * min(ankle_vis, hip_vis) * np.clip(margin, 0.3, 1))
    else:
        quiet = sig.ankle_speed[:peak] / cfg["stance_ankle_speed"] + sig.hip_speed[:peak] / cfg["stance_hip_speed"]
        fallback = int(np.argmin(quiet)) if peak > 0 else 0
        stance = moment(fallback, 0.3 * stride_clarity * min(ankle_vis, hip_vis))
        notes.append("Batter never settled before the stride; stance is the quietest frame.")

    # --- FRONT_FOOT_LANDING -----------------------------------------------------------
    fwd_limit = max(cfg["landing_fwd_abs"], cfg["landing_fwd_rel"] * peak_speed)
    settled = (sig.ankle_forward < fwd_limit) & (np.abs(sig.ankle_vertical) < cfg["landing_vert_speed"])
    start = _first_run(settled, peak + 1, cfg["landing_hold_frames"])
    if start is not None:
        # Cleaner stop (lower residual speed) -> higher confidence.
        residual = float(np.hypot(sig.ankle_forward[start], sig.ankle_vertical[start]))
        landing = moment(start, stride_clarity * ankle_vis * np.clip(1 - residual / fwd_limit * 0.5, 0.5, 1))
    elif peak + 1 < n:
        fallback = peak + 1 + int(np.argmin(sig.ankle_speed[peak + 1:]))
        landing = moment(fallback, 0.3 * stride_clarity * ankle_vis)
        notes.append("Front foot never clearly settled; landing is the slowest frame after the stride.")
    else:
        landing = None
        notes.append("Clip ends at the stride peak; no landing found.")

    # --- CONTACT ----------------------------------------------------------------------
    contact = None
    if landing is not None:
        lo = max(0, landing.frame - round(cfg["contact_pre_s"] * fps))
        hi = min(n, landing.frame + round(cfg["contact_window_s"] * fps) + 1)
        window = sig.wrist_forward[lo:hi]
        best = int(np.argmax(window))
        best_speed = float(window[best])
        if best_speed < cfg["min_wrist_speed"]:
            notes.append("No clear forward hand movement around landing; contact not found.")
        else:
            # Contact cannot precede landing: look for the hands arriving from there on.
            start = max(lo + best, landing.frame)
            slowed = np.flatnonzero(sig.wrist_forward[start:] < cfg["contact_release_fraction"] * best_speed)
            # Prominence: how far the peak stands above the typical speed in the window.
            prominence = 1 - max(0.0, float(np.median(window))) / best_speed
            penalty = 0.6 if best in (0, len(window) - 1) else 1.0
            if len(slowed):
                frame = start + int(slowed[0])
            else:
                frame, penalty = start, penalty * 0.5
                notes.append("Hands never slowed down after the peak; contact is only a rough guess.")
            if best in (0, len(window) - 1):
                notes.append("Hand-speed peak is at the edge of the search window.")
            contact = moment(frame, prominence * penalty * wrist_vis * (0.5 + 0.5 * landing.confidence))

    return KeyMoments(stance, landing, contact, sig, tuple(notes))

