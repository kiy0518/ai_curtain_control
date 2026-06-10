"""Arm-based gestures from COCO 17-keypoint body pose (for far-range control).

Deliberate poses (avoid natural standing/arms-down false triggers):
  * OPEN  (열림): 양팔 위로(만세) — both wrists clearly above shoulders
  * CLOSE (닫힘): 양팔 가슴 앞 교차(X) — both wrists between shoulders, chest height
  * STOP  (정지): 한 팔 수평(T) — a wrist near shoulder height, extended sideways

Scale-invariant: thresholds are relative to shoulder width, so they work at
different distances. Uses keypoint confidence to ignore unseen joints.
"""

import numpy as np

SH_L, SH_R = 5, 6
WR_L, WR_R = 9, 10
_CONF = 0.3


def classify(kp):
    """kp: (17,3) array of (x, y, conf). Returns OPEN/CLOSE/STOP or None."""
    lsh, lsh_c = kp[SH_L, :2], kp[SH_L, 2]
    rsh, rsh_c = kp[SH_R, :2], kp[SH_R, 2]
    if lsh_c < _CONF or rsh_c < _CONF:
        return None                              # need both shoulders

    sw = abs(lsh[0] - rsh[0]) + 1e-6             # shoulder width = scale unit
    sh_y = (lsh[1] + rsh[1]) / 2.0
    x_lo, x_hi = min(lsh[0], rsh[0]), max(lsh[0], rsh[0])
    v_margin = 0.3 * sw

    lw, lw_c = kp[WR_L, :2], kp[WR_L, 2]
    rw, rw_c = kp[WR_R, :2], kp[WR_R, 2]

    l_up = lw_c > _CONF and lw[1] < lsh[1] - v_margin   # y smaller = higher
    r_up = rw_c > _CONF and rw[1] < rsh[1] - v_margin

    # 양팔 위로 → 열림
    if l_up and r_up:
        return "OPEN"

    # 양팔 가슴 앞 교차 → 닫힘 (두 손목이 어깨 사이 + 가슴 높이)
    if lw_c > _CONF and rw_c > _CONF:
        chest = abs(lw[1] - sh_y) < 0.6 * sw and abs(rw[1] - sh_y) < 0.6 * sw
        between = (x_lo < lw[0] < x_hi) and (x_lo < rw[0] < x_hi)
        if chest and between:
            return "CLOSE"

    # 한 팔 수평(T) → 정지 (손목이 어깨 높이 + 옆으로 뻗음)
    def horiz(w, wc, sh):
        return wc > _CONF and abs(w[1] - sh[1]) < v_margin and \
            abs(w[0] - sh[0]) > 0.8 * sw
    if horiz(lw, lw_c, lsh) or horiz(rw, rw_c, rsh):
        return "STOP"

    return None
