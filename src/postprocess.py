"""Decode the split-branch YOLOv8-pose RKNN output into hand detections.

This matches the Rockchip ``rknn_model_zoo`` ``yolov8_pose`` export, where the
model's post-process (DFL + box/grid decode) is stripped out and run here on
CPU for better NPU quantisation and speed.

Model outputs (order-independent — we sort them by shape):
  * 3 detection branches, each ``(1, DET_CHANNELS, H, W)`` for strides 8/16/32
        channels = 4*REG_MAX box-DFL logits  +  NUM_CLASSES class logits
  * 1 keypoint branch ``(1, KPT_CHANNELS, total_anchors)``
        channels = NUM_KEYPOINTS * 3  (x, y in letterbox px; visibility logit)

``decode`` returns detections in original-frame coordinates, matching the
schema ``draw.draw_detections`` expects:
    {"box": (x1,y1,x2,y2) int, "score": float, "keypoints": (21,3) float}
"""

import numpy as np

from constants import (NUM_KEYPOINTS, NUM_CLASSES, REG_MAX, INPUT_SIZE,
                       DET_CHANNELS, KPT_CHANNELS)

_BINS = np.arange(REG_MAX, dtype=np.float32)

# Standard (decoded) ultralytics pose output channel count:
#   4 (xywh) + NUM_CLASSES + NUM_KEYPOINTS*3
STD_CHANNELS = 4 + NUM_CLASSES + KPT_CHANNELS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _nms(boxes, scores, iou_thres):
    """NumPy NMS on xyxy boxes; returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def _split_outputs(outputs):
    """Sort raw model outputs into (detection branches, keypoint branch)."""
    det, kpt = [], None
    for o in outputs:
        o = np.asarray(o)
        ch = o.shape[1]
        if ch == KPT_CHANNELS:
            kpt = o.reshape(1, KPT_CHANNELS, -1)
        elif ch == DET_CHANNELS:
            det.append(o)
        # any other branch (e.g. score-sum) is ignored
    # stride 8 first (largest feature map) -> matches keypoint anchor order
    det.sort(key=lambda o: o.shape[2] * o.shape[3], reverse=True)
    return det, kpt


def _decode_standard(out, letterbox, conf_thres, iou_thres):
    """Decode a single already-decoded output ``(1, 68, N)`` / ``(1, N, 68)``.

    This is what the *standard* Ultralytics ONNX export produces (box xywh +
    class conf + keypoints, all decoded in-graph). Used as a fallback when the
    model wasn't exported with the airockchip split head.
    """
    pred = np.squeeze(np.asarray(out))
    if pred.shape[0] != STD_CHANNELS:        # ensure channels-first (68, N)
        pred = pred.transpose()
    pred = pred.transpose()                   # -> (N, 68), one row per anchor

    scores = pred[:, 4]
    mask = scores >= conf_thres
    pred, scores = pred[mask], scores[mask]
    if pred.shape[0] == 0:
        return []

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kpts = pred[:, 5:].reshape(-1, NUM_KEYPOINTS, 3)

    ratio, pad_x, pad_y = letterbox
    detections = []
    for i in _nms(boxes, scores, iou_thres):
        box = boxes[i].copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / ratio
        box[[1, 3]] = (box[[1, 3]] - pad_y) / ratio
        kp = kpts[i].copy()
        kp[:, 0] = (kp[:, 0] - pad_x) / ratio
        kp[:, 1] = (kp[:, 1] - pad_y) / ratio
        detections.append({"box": box.astype(np.int32),
                           "score": float(scores[i]), "keypoints": kp})
    return detections


def decode(outputs, letterbox, conf_thres=0.5, iou_thres=0.45):
    # Auto-detect output format: a single (1,68,N)/(1,N,68) tensor means the
    # model was exported with the standard (decoded) head; otherwise it's the
    # airockchip split-branch head (3 det + 1 kpt).
    arrs = [np.asarray(o) for o in outputs]
    for o in arrs:
        sq = np.squeeze(o)
        if sq.ndim == 2 and STD_CHANNELS in sq.shape:
            return _decode_standard(o, letterbox, conf_thres, iou_thres)

    ratio, pad_x, pad_y = letterbox
    det_branches, kpt_out = _split_outputs(outputs)

    boxes_all, scores_all, kpts_all = [], [], []
    anchor_offset = 0
    for branch in det_branches:
        _, _, H, W = branch.shape
        stride = INPUT_SIZE // H
        feat = branch.reshape(DET_CHANNELS, H * W)
        box_dist = feat[:4 * REG_MAX, :]                 # (64, N)
        cls = _sigmoid(feat[4 * REG_MAX:, :])            # (nc, N)
        cls_score = cls.max(axis=0)
        keep = np.where(cls_score >= conf_thres)[0]

        if keep.size:
            # DFL: softmax over the 16 bins of each of the 4 box sides
            d = box_dist[:, keep].reshape(4, REG_MAX, -1)
            d = _softmax(d, axis=1)
            ltrb = (d * _BINS[None, :, None]).sum(axis=1)  # (4, n)
            gx = (keep % W) + 0.5
            gy = (keep // W) + 0.5
            x1 = (gx - ltrb[0]) * stride
            y1 = (gy - ltrb[1]) * stride
            x2 = (gx + ltrb[2]) * stride
            y2 = (gy + ltrb[3]) * stride
            boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))
            scores_all.append(cls_score[keep])

            if kpt_out is not None:
                cols = anchor_offset + keep
                kp = kpt_out[0, :, cols].T.reshape(-1, NUM_KEYPOINTS, 3)
                kp = kp.astype(np.float32).copy()
                kp[:, :, 2] = _sigmoid(kp[:, :, 2])      # visibility -> [0,1]
                kpts_all.append(kp)

        anchor_offset += H * W

    if not boxes_all:
        return []

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    kpts = (np.concatenate(kpts_all, axis=0) if kpts_all
            else np.zeros((len(boxes), NUM_KEYPOINTS, 3), np.float32))

    detections = []
    for i in _nms(boxes, scores, iou_thres):
        box = boxes[i].copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / ratio
        box[[1, 3]] = (box[[1, 3]] - pad_y) / ratio

        kp = kpts[i].copy()
        kp[:, 0] = (kp[:, 0] - pad_x) / ratio
        kp[:, 1] = (kp[:, 1] - pad_y) / ratio

        detections.append({
            "box": box.astype(np.int32),
            "score": float(scores[i]),
            "keypoints": kp,
        })
    return detections
