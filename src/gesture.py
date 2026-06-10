"""Map 21 hand keypoints to a curtain-control gesture.

Gestures (chosen to be visually distinct and orientation-robust):
  * OPEN  (열림)  : open palm  — index, middle, ring, pinky all extended
  * CLOSE (닫힘)  : fist       — those four fingers all curled
  * STOP  (정지)  : V sign     — index + middle extended, ring + pinky curled

A finger is "extended" when its tip is farther from the wrist than its PIP
joint (distance-based, so it works regardless of hand rotation). The thumb is
ignored — it is the least reliable and not needed to separate these three.
"""

import numpy as np

from constants import GESTURE_KR  # noqa: F401  (single source; re-export)

WRIST = 0
# finger -> (tip index, pip/reference joint index)
_FINGERS = {
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}


def extended_fingers(kpts):
    """Return dict {finger: bool extended} for index/middle/ring/pinky."""
    w = kpts[WRIST, :2]
    out = {}
    for name, (tip, pip) in _FINGERS.items():
        out[name] = (np.linalg.norm(kpts[tip, :2] - w) >
                     np.linalg.norm(kpts[pip, :2] - w))
    return out


def classify(kpts):
    """Return 'OPEN' / 'CLOSE' / 'STOP' or None for an unrecognised pose.

    Thresholds are tolerant to a single mis-detected finger (the model's
    curled-finger keypoints are noisy)."""
    st = extended_fingers(kpts)
    i, m, r, p = st["index"], st["middle"], st["ring"], st["pinky"]
    n = sum((i, m, r, p))
    if n >= 4:
        return "OPEN"
    if n <= 1:
        return "CLOSE"
    if i and m and not r and not p:
        return "STOP"
    return None


class GestureStabilizer:
    """Debounce raw per-frame labels: a label must repeat for ``hold`` frames
    before it is committed. Prevents the banner from flickering."""

    def __init__(self, hold=5):
        self.hold = hold
        self._cand = None
        self._count = 0
        self.committed = None

    def update(self, label):
        if label == self._cand:
            self._count += 1
        else:
            self._cand = label
            self._count = 1
        if self._count >= self.hold:
            self.committed = self._cand
        return self.committed
