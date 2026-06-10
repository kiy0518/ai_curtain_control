"""RKNNLite wrapper that runs the hand-pose model on the RK3576 NPU."""

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from camera import letterbox
from postprocess import decode
from constants import INPUT_SIZE


class HandPose:
    """Load a ``.rknn`` hand-pose model and run inference on frames."""

    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45,
                 core_mask=RKNNLite.NPU_CORE_AUTO, imgsz=INPUT_SIZE):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        # RK3576 has multiple NPU cores; AUTO lets the runtime schedule.
        if self.rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("Failed to init RKNN runtime")

    def infer(self, frame_bgr):
        """Run detection on a BGR frame; return list of hand detections."""
        padded, lb = letterbox(frame_bgr, self.imgsz)
        # RKNN expects RGB, NHWC, uint8 (quantization handled internally).
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb, axis=0)

        # Split-branch model -> 4 outputs; decode() sorts them by shape.
        outputs = self.rknn.inference(inputs=[inp])
        return decode(outputs, lb, self.conf_thres, self.iou_thres)

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
