"""Draw hand detections (boxes, skeleton, keypoints) onto a frame."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constants import HAND_SKELETON, FINGERTIPS, GESTURE_KR

_KP_CONF_THRES = 0.3
_BOX_COLOR = (180, 105, 255)   # pink (BGR) for bounding box + label

# Korean-capable font for the gesture banner (cv2 can't render Hangul).
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_font_cache = {}

# Banner colours per gesture (RGB, for PIL).
_GESTURE_RGB = {"OPEN": (40, 220, 60), "CLOSE": (235, 60, 60), "STOP": (255, 180, 0)}


def _font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
    return _font_cache[size]


def draw_gesture_banner(frame, label):
    """Render the Korean gesture name (열림/닫힘/정지) centred at top of frame."""
    if not label:
        return frame
    text = GESTURE_KR.get(label, label)
    rgb = _GESTURE_RGB.get(label, (255, 255, 255))

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    size = max(32, frame.shape[1] // 14)
    font = _font(size)

    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    cx = frame.shape[1] // 2
    pad = size // 4
    bx0, by0 = cx - tw // 2 - pad, 10
    bx1, by1 = cx + tw // 2 + pad, 10 + th + 2 * pad
    d.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
    d.text((cx - tw // 2 - x0, by0 + pad - y0), text, font=font, fill=rgb)

    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def draw_detections(frame, detections, skeleton=HAND_SKELETON,
                    highlight=FINGERTIPS, label="hand"):
    """Render detections in place. ``skeleton``/``highlight`` come from the
    active model profile (hand vs body)."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        cv2.putText(frame, f"{label} {det['score']:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR, 1, cv2.LINE_AA)

        kpts = det["keypoints"]
        for a, b in skeleton:
            if kpts[a, 2] < _KP_CONF_THRES or kpts[b, 2] < _KP_CONF_THRES:
                continue
            cv2.line(frame, (int(kpts[a, 0]), int(kpts[a, 1])),
                     (int(kpts[b, 0]), int(kpts[b, 1])), (255, 160, 0), 2, cv2.LINE_AA)

        for idx in range(kpts.shape[0]):
            if kpts[idx, 2] < _KP_CONF_THRES:
                continue
            color = (0, 0, 255) if idx in highlight else (0, 255, 255)
            cv2.circle(frame, (int(kpts[idx, 0]), int(kpts[idx, 1])), 3, color, -1)

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"{fps:.1f} FPS", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
    return frame
