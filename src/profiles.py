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
from typing import Callable, List, Optional, Tuple

import constants
import gesture as gesture_hand
import gesture_body
import gesture_motion


@dataclass
class ModelProfile:
    name: str
    model_path: str
    imgsz: int
    num_keypoints: int
    num_classes: int
    skeleton: List[Tuple[int, int]]
    highlight: List[int]            # keypoint indices to emphasise
    # 정적 분류기: (kp (K,3)) -> 'OPEN'/'CLOSE'/'STOP'/None — 스테이트리스,
    # GestureStabilizer(hold N프레임)로 디바운싱.
    classify: Optional[Callable] = None
    # 이벤트형 분류기 팩토리: () -> obj with .update(dets, now) — 상태를 가지며
    # 제스처 확정 순간 1회만 라벨 반환(자체 디바운싱). 둘 중 하나만 설정.
    make_classifier: Optional[Callable] = None
    # 적응형 신뢰도(작은 박스=관대) 크기 기준 — 프로파일마다 다름:
    # 박스높이/프레임높이 ≤ dyn_small_h → floor(=conf*dyn_floor_ratio),
    # ≥ dyn_big_h → full conf. 손은 작게(0.12~0.45), 전신은 세로로 길어 크게.
    dyn_small_h: float = 0.12
    dyn_big_h: float = 0.45
    dyn_floor_ratio: float = 0.4
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
        dyn_small_h=0.30, dyn_big_h=0.80,   # 사람 박스는 세로로 김 → 기준 키움
        desc="원거리 전신 팔 제스처: 오른팔수평=열림 / 왼팔수평=닫힘 / X교차=정지",
    ),
    # 원거리: 전신 17키포인트, 손목 '움직임' 제스처 (모델은 body_far와 동일)
    "body_motion": ModelProfile(
        name="body_motion",
        model_path="models/body_pose_640.rknn",
        imgsz=640,
        num_keypoints=17,
        num_classes=1,
        skeleton=constants.BODY_SKELETON,
        highlight=[9, 10],          # wrists
        make_classifier=gesture_motion.WristMotionClassifier,
        dyn_small_h=0.30, dyn_big_h=0.80,   # 사람 박스는 세로로 김 → 기준 키움
        desc="원거리 손목 움직임: 손 들고 우→좌=열림 / 좌→우=닫힘 / 멈춤유지=정지",
    ),
}


def get_profile(name):
    if name not in PROFILES:
        raise KeyError(f"unknown profile '{name}'. available: {list(PROFILES)}")
    return PROFILES[name]
