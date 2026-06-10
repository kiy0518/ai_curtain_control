"""Map 21 hand keypoints to a curtain-control gesture.

Gestures:
  * OPEN  (열림)  : 👍 thumb out + UP,   other 4 fingers folded
  * CLOSE (닫힘)  : 👈 thumb out + SIDE, other 4 fingers folded
  * STOP  (정지)  : 🖐 open palm — all 4 fingers (index/middle/ring/pinky) extended

A finger is "extended" when its tip is farther from the wrist than its PIP
joint. The thumb is handled separately (out-or-not + up-vs-side direction).
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

THUMB_TIP, MIDDLE_MCP = 4, 9
_FINGER_MCPS = (5, 9, 13, 17)   # index/middle/ring/pinky knuckles


def _thumb_pose(kpts):
    """Classify the thumb-tip position w.r.t. the fist: 'up' / 'side' / None.

    Uses the knuckle line (finger MCPs) and palm length as scale. Robust to a
    plain fist (thumb tucked) → returns None (no false OPEN/CLOSE).
    """
    wrist = kpts[WRIST, :2]
    scale = np.linalg.norm(kpts[MIDDLE_MCP, :2] - wrist) + 1e-6   # palm length
    tip = kpts[THUMB_TIP, :2]
    knuckles_y = min(kpts[i, 1] for i in _FINGER_MCPS)            # top of fist
    if tip[1] < knuckles_y - 0.25 * scale:                        # 👍 above fist
        return "up"
    if abs(tip[0] - wrist[0]) > 0.55 * scale:                     # 👈 far to side
        return "side"
    return None


def extended_fingers(kpts):
    """Return dict {finger: bool extended} for index/middle/ring/pinky."""
    w = kpts[WRIST, :2]
    out = {}
    for name, (tip, pip) in _FINGERS.items():
        out[name] = (np.linalg.norm(kpts[tip, :2] - w) >
                     np.linalg.norm(kpts[pip, :2] - w))
    return out


def classify(kpts):
    """Return 'OPEN' / 'CLOSE' / 'STOP' or None.

    🖐 open palm(4손가락 펴짐) → STOP
    👍 주먹+엄지 위  → OPEN
    👈 주먹+엄지 옆  → CLOSE
    """
    st = extended_fingers(kpts)
    n = sum(st.values())            # index/middle/ring/pinky extended count

    if n >= 4:                      # open palm
        return "STOP"
    if n <= 1:                      # fist — thumb direction decides
        pose = _thumb_pose(kpts)
        if pose == "up":
            return "OPEN"
        if pose == "side":
            return "CLOSE"
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
