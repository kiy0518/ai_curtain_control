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
from constants import INPUT_SIZE, NUM_KEYPOINTS, NUM_CLASSES


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

    def infer(self, frame_bgr):
        padded, lb = letterbox(frame_bgr, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)   # RKNN expects RGB NHWC uint8
        inp = np.expand_dims(rgb, axis=0)
        outputs = self.rknn.inference(inputs=[inp])
        return decode(outputs, lb, self.num_keypoints, self.num_classes,
                      self.conf_thres, self.iou_thres)

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


PoseModel = HandPose
