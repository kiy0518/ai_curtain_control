"""RKNNLite wrapper that runs a YOLOv8-pose model on the RK3576 NPU.

Generic over keypoint topology (hand 21 / body 17) via ``num_keypoints``/
``num_classes`` — driven by a model profile (see ``profiles.py``). The legacy
name ``HandPose`` is kept; ``PoseModel`` is an alias.
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from camera import letterbox
from postprocess import decode
from constants import INPUT_SIZE, NUM_KEYPOINTS, NUM_CLASSES, HAND_SKELETON


# 뼈길이/박스대각 비가 이 이상이면 키포인트 뒤섞임(잘못된 chirality) 의심
# (실측: 정상 ~2.3, 스크램블 ~3.4) → 조건부 TTA 트리거 임계
CHIRALITY_RATIO = 2.8


def _bone_len(kp, skeleton):
    """스켈레톤 뼈 길이 합 — 키포인트가 올바르면 작고, 뒤섞이면(교차) 크다."""
    return float(sum(np.hypot(kp[a, 0] - kp[b, 0], kp[a, 1] - kp[b, 1])
                     for a, b in skeleton))


def _plausible(d, skeleton):
    """손 키포인트가 그럴듯한가 = 뼈길이 합이 박스 대각선 대비 과대하지 않은가."""
    b = d["box"]
    diag = float(np.hypot(b[2] - b[0], b[3] - b[1])) + 1e-6
    return _bone_len(d["keypoints"], skeleton) / diag <= CHIRALITY_RATIO


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6
    return inter / ua


def _merge_chirality(a, b, skeleton):
    """같은 손은 a(원본)/b(좌우반전복원) 중 뼈길이가 작은(더 그럴듯한) 쪽 채택."""
    out, used = [], set()
    for da in a:
        bj, biou = -1, 0.3
        for j, db in enumerate(b):
            if j in used:
                continue
            v = _iou(da["box"], db["box"])
            if v > biou:
                biou, bj = v, j
        if bj >= 0:
            used.add(bj)
            db = b[bj]
            out.append(da if _bone_len(da["keypoints"], skeleton)
                       <= _bone_len(db["keypoints"], skeleton) else db)
        else:
            out.append(da)
    out.extend(db for j, db in enumerate(b) if j not in used)
    return out


class HandPose:
    """Load a ``.rknn`` pose model and run inference on BGR frames."""

    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45,
                 core_mask=RKNNLite.NPU_CORE_AUTO, imgsz=INPUT_SIZE,
                 num_keypoints=NUM_KEYPOINTS, num_classes=NUM_CLASSES):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        # 손(21점)은 모델 좌우(chirality) 편향이 있어 양방향 추론으로 보정(TTA).
        self.tta = (num_keypoints == 21)
        self.skeleton = HAND_SKELETON

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        if self.rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("Failed to init RKNN runtime")

    @classmethod
    def from_profile(cls, profile, conf_thres=0.5, iou_thres=0.45,
                     model_path=None, imgsz=None):
        return cls(model_path or profile.model_path,
                   conf_thres=conf_thres, iou_thres=iou_thres,
                   imgsz=imgsz or profile.imgsz,
                   num_keypoints=profile.num_keypoints,
                   num_classes=profile.num_classes)

    def _run(self, frame_bgr):
        padded, lb = letterbox(frame_bgr, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)   # RKNN expects RGB NHWC uint8
        outputs = self.rknn.inference(inputs=[np.expand_dims(rgb, axis=0)])
        return decode(outputs, lb, self.num_keypoints, self.num_classes,
                      self.conf_thres, self.iou_thres)

    def infer(self, frame_bgr):
        a = self._run(frame_bgr)
        # 조건부 TTA: 모든 손이 그럴듯하면 1회 추론으로 끝(FPS 유지).
        # 뒤섞인(chirality 오류) 손이 있을 때만 좌우반전 추론 추가(2x).
        if not self.tta or not a or all(_plausible(d, self.skeleton) for d in a):
            return a
        # 좌우반전본도 추론(모델이 잘 맞추는 손모양으로 만들어줌) → 좌표 복원 후 병합
        H, W = frame_bgr.shape[:2]
        b = self._run(cv2.flip(frame_bgr, 1))
        for d in b:
            x1, y1, x2, y2 = d["box"]
            d["box"] = np.array([W - 1 - x2, y1, W - 1 - x1, y2], d["box"].dtype)
            d["keypoints"][:, 0] = (W - 1) - d["keypoints"][:, 0]
        return _merge_chirality(a, b, self.skeleton)

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


PoseModel = HandPose
