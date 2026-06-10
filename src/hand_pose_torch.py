"""CPU inference backend using Ultralytics directly on a .pt model.

This is the no-conversion fallback: it runs the hand-pose model on the board's
CPU via PyTorch. Slower than the RKNN/NPU path but needs no x86 conversion, so
it gets the full pipeline working immediately. Returns the same detection dict
schema as ``hand_pose.HandPose`` so ``draw.draw_detections`` is unchanged.

For the fast NPU path, convert the model to .rknn and use ``hand_pose.HandPose``.
"""

import numpy as np


class HandPoseTorch:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

    def infer(self, frame_bgr):
        """Run detection on a BGR frame; return list of hand detections.

        Ultralytics handles letterbox + decode internally and returns
        keypoints/boxes already in original-frame pixel coordinates.
        """
        results = self.model(
            frame_bgr, imgsz=self.imgsz, conf=self.conf_thres,
            iou=self.iou_thres, verbose=False)
        r = results[0]

        dets = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        if r.keypoints is not None:
            kpts = r.keypoints.data.cpu().numpy()        # (N, 21, 3) -> x, y, conf
        else:
            kpts = np.zeros((len(boxes), 21, 3), np.float32)

        for i in range(len(boxes)):
            dets.append({
                "box": boxes[i].astype(np.int32),
                "score": float(confs[i]),
                "keypoints": kpts[i].astype(np.float32),
            })
        return dets

    def release(self):
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()