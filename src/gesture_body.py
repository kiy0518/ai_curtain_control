"""Arm-based gestures from COCO 17-keypoint body pose (for far-range control).

Deliberate poses (avoid natural standing/arms-down false triggers):
  * OPEN  (열림): 오른팔 수평 — right wrist extended sideways at shoulder height
  * CLOSE (닫힘): 왼팔 수평  — left wrist extended sideways at shoulder height
  * STOP  (정지): 양팔 X 교차 — both wrists between shoulders, chest height

Note: L/R follow COCO (the person's own left/right). If it feels mirrored on
screen, swap WR_L/WR_R below. Scale-invariant via shoulder width.
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
    v_margin = 0.35 * sw

    lw, lw_c = kp[WR_L, :2], kp[WR_L, 2]
    rw, rw_c = kp[WR_R, :2], kp[WR_R, 2]

    # 양팔 X 교차 → 정지 (두 손목이 어깨 사이 + 가슴 높이)
    if lw_c > _CONF and rw_c > _CONF:
        chest = abs(lw[1] - sh_y) < 0.6 * sw and abs(rw[1] - sh_y) < 0.6 * sw
        between = (x_lo < lw[0] < x_hi) and (x_lo < rw[0] < x_hi)
        if chest and between:
            return "STOP"

    # 한 팔 수평 → 손목이 어깨 높이 + 바깥으로 뻗음
    def horiz(w, wc, sh):
        return (wc > _CONF and abs(w[1] - sh[1]) < v_margin and
                abs(w[0] - sh[0]) > 0.8 * sw)

    r_horiz = horiz(rw, rw_c, rsh)
    l_horiz = horiz(lw, lw_c, lsh)
    if r_horiz and not l_horiz:
        return "OPEN"                            # 오른팔 수평 = 열림
    if l_horiz and not r_horiz:
        return "CLOSE"                           # 왼팔 수평 = 닫힘

    return None
