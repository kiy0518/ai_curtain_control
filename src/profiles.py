"""Selectable model profiles.

A profile bundles everything that changes when you swap the AI model:
the .rknn path, input size, keypoint topology (count + skeleton), which points
to highlight, and the gesture classifier. Switching profiles switches the whole
interpretation — so the same pipeline serves close-range *hand* control and
far-range *body* control.

Add/edit profiles here; the dashboard "model selection" (Phase 2) will expose
them. ``serve.py --profile <name>`` selects one (path/imgsz overridable).
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import constants
import gesture as gesture_hand
import gesture_body


@dataclass
class ModelProfile:
    name: str
    model_path: str
    imgsz: int
    num_keypoints: int
    num_classes: int
    skeleton: List[Tuple[int, int]]
    highlight: List[int]            # keypoint indices to emphasise
    classify: Callable              # (kp (K,3)) -> 'OPEN'/'CLOSE'/'STOP'/None
    desc: str = ""


PROFILES = {
    # 근거리: 손 21키포인트, 손가락 제스처 (열림=손바닥/닫힘=주먹/정지=브이)
    "hand_near": ModelProfile(
        name="hand_near",
        model_path="models/hand_pose_640.rknn",
        imgsz=640,
        num_keypoints=21,
        num_classes=1,
        skeleton=constants.HAND_SKELETON,
        highlight=constants.FINGERTIPS,
        classify=gesture_hand.classify,
        desc="근거리 손 제스처: 👈엄지 옆=열림 / 👍엄지 위=닫힘 / 🖐손바닥=정지",
    ),
    # 원거리: 전신 17키포인트, 팔 제스처 (양팔위=열림/교차=닫힘/한팔수평=정지)
    "body_far": ModelProfile(
        name="body_far",
        model_path="models/body_pose_640.rknn",
        imgsz=640,
        num_keypoints=17,
        num_classes=1,
        skeleton=constants.BODY_SKELETON,
        highlight=[9, 10],          # wrists
        classify=gesture_body.classify,
        desc="원거리 전신 팔 제스처: 양팔위=열림 / 가슴앞교차=닫힘 / 한팔수평=정지",
    ),
}


def get_profile(name):
    if name not in PROFILES:
        raise KeyError(f"unknown profile '{name}'. available: {list(PROFILES)}")
    return PROFILES[name]
