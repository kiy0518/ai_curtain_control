"""Map 21 hand keypoints to a curtain-control gesture.

Gestures (thumb-direction scheme; open palm does nothing):
  * STOP  (정지)  : 👍 thumb UP (fist)
  * OPEN  (열림)  : 👈/👉 thumb to one SIDE, fist  (which side = swap_lr)
  * CLOSE (닫힘)  : 👉/👈 thumb to the OTHER side, fist

A finger is "extended" when its tip is farther from the wrist than its PIP
joint. The thumb is handled separately (up vs left vs right).

The LEFT↔RIGHT → OPEN/CLOSE mapping is flipped by ``set_swap_lr(True)`` (a web
toggle), since which way "feels" like open depends on camera mirroring/mounting.
"""

import numpy as np

from constants import GESTURE_KR  # noqa: F401  (single source; re-export)

WRIST = 0

# 엄지 좌/우 → 열림/닫힘 매핑 반전 플래그(웹에서 토글). 거울/장착방향에 따라
# 어느 쪽이 '열림'으로 느껴지는지 달라서 사용자가 뒤집을 수 있게 함.
_SWAP_LR = False


def set_swap_lr(v):
    """엄지 좌우 ↔ 열림/닫힘 매핑을 반전(True)/정상(False)으로 설정."""
    global _SWAP_LR
    _SWAP_LR = bool(v)


def get_swap_lr():
    return _SWAP_LR
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
    """Classify the thumb-tip position w.r.t. the fist: 'up'/'left'/'right'/None.

    Uses the knuckle line (finger MCPs) and palm length as scale. Robust to a
    plain fist (thumb tucked) → returns None (no false gesture). 'up' is checked
    first so a raised thumb is never mistaken for a side thumb.
    """
    wrist = kpts[WRIST, :2]
    scale = np.linalg.norm(kpts[MIDDLE_MCP, :2] - wrist) + 1e-6   # palm length
    tip = kpts[THUMB_TIP, :2]
    knuckles_y = min(kpts[i, 1] for i in _FINGER_MCPS)            # top of fist
    if tip[1] < knuckles_y - 0.25 * scale:                        # 👍 above fist
        return "up"
    dx = tip[0] - wrist[0]
    if abs(dx) > 0.55 * scale:                                    # 👈/👉 to a side
        return "right" if dx > 0 else "left"
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

    👍 주먹+엄지 위  → STOP (정지)
    👈/👉 주먹+엄지 옆 → OPEN/CLOSE (좌우 매핑은 set_swap_lr 로 반전)
    (🖐 손바닥 펼침은 아무 동작 안 함)
    """
    st = extended_fingers(kpts)
    n = sum(st.values())            # index/middle/ring/pinky extended count

    if n <= 1:                      # fist — thumb direction decides
        pose = _thumb_pose(kpts)
        if pose == "up":
            return "STOP"           # 👍 엄지 위 = 정지
        if pose == "left":
            return "CLOSE" if _SWAP_LR else "OPEN"
        if pose == "right":
            return "OPEN" if _SWAP_LR else "CLOSE"
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

    @property
    def candidate(self):
        return self._cand

    @property
    def count(self):
        return self._count
