"""Decode YOLOv8-pose RKNN output into detections (hand or body).

Parameterised by ``num_keypoints``/``num_classes`` so the same code serves the
21-keypoint hand model and the 17-keypoint body model. Handles BOTH export
formats automatically:

* **split-branch** (Rockchip rknn_model_zoo / airockchip): 3 detection branches
  ``(1, 4*REG_MAX+nc, H, W)`` + 1 keypoint branch ``(1, nkpt*3, anchors)``,
  with DFL/box decode done here on CPU.
* **standard** (plain ultralytics ONNX): a single decoded tensor
  ``(1, 4+nc+nkpt*3, anchors)``.

Returns detections in original-frame coords:
    {"box": (x1,y1,x2,y2) int, "score": float, "keypoints": (K,3) float}
"""

import numpy as np

from constants import REG_MAX, NUM_KEYPOINTS, NUM_CLASSES

_BINS = np.arange(REG_MAX, dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _nms(boxes, scores, iou_thres):
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


def _map_back(box, kp, letterbox):
    """Map letterbox-space box/keypoints back to original frame coords."""
    ratio, pad_x, pad_y = letterbox
    box = box.copy()
    box[[0, 2]] = (box[[0, 2]] - pad_x) / ratio
    box[[1, 3]] = (box[[1, 3]] - pad_y) / ratio
    kp = kp.copy()
    kp[:, 0] = (kp[:, 0] - pad_x) / ratio
    kp[:, 1] = (kp[:, 1] - pad_y) / ratio
    return box.astype(np.int32), kp


def _decode_standard(out, letterbox, num_keypoints, num_classes,
                     conf_thres, iou_thres):
    std_ch = 4 + num_classes + num_keypoints * 3
    pred = np.squeeze(np.asarray(out))
    if pred.shape[0] != std_ch:
        pred = pred.transpose()
    pred = pred.transpose()                       # (N, std_ch)

    # class score = max over class channels (single class -> channel 4)
    scores = pred[:, 4:4 + num_classes].max(axis=1)
    mask = scores >= conf_thres
    pred, scores = pred[mask], scores[mask]
    if pred.shape[0] == 0:
        return []

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kpts = pred[:, 4 + num_classes:].reshape(-1, num_keypoints, 3)

    dets = []
    for i in _nms(boxes, scores, iou_thres):
        box, kp = _map_back(boxes[i], kpts[i], letterbox)
        dets.append({"box": box, "score": float(scores[i]), "keypoints": kp})
    return dets


def _decode_split(outputs, letterbox, num_keypoints, num_classes,
                  conf_thres, iou_thres):
    det_ch = 4 * REG_MAX + num_classes
    kpt_ch = num_keypoints * 3

    det, kpt_out = [], None
    for o in outputs:
        o = np.asarray(o)
        if o.shape[1] == kpt_ch:
            kpt_out = o.reshape(1, kpt_ch, -1)
        elif o.shape[1] == det_ch:
            det.append(o)
    det.sort(key=lambda o: o.shape[2] * o.shape[3], reverse=True)
    if not det:
        return []
    imgsz = det[0].shape[2] * 8                   # stride-8 branch -> imgsz

    boxes_all, scores_all, kpts_all = [], [], []
    anchor_offset = 0
    for branch in det:
        _, _, H, W = branch.shape
        stride = imgsz // H
        feat = branch.reshape(det_ch, H * W)
        box_dist = feat[:4 * REG_MAX, :]
        cls_score = _sigmoid(feat[4 * REG_MAX:, :]).max(axis=0)
        keep = np.where(cls_score >= conf_thres)[0]
        if keep.size:
            d = _softmax(box_dist[:, keep].reshape(4, REG_MAX, -1), axis=1)
            ltrb = (d * _BINS[None, :, None]).sum(axis=1)
            gx = (keep % W) + 0.5
            gy = (keep // W) + 0.5
            boxes_all.append(np.stack([
                (gx - ltrb[0]) * stride, (gy - ltrb[1]) * stride,
                (gx + ltrb[2]) * stride, (gy + ltrb[3]) * stride], axis=1))
            scores_all.append(cls_score[keep])
            if kpt_out is not None:
                kp = kpt_out[0, :, anchor_offset + keep].T.reshape(
                    -1, num_keypoints, 3).astype(np.float32).copy()
                kp[:, :, 2] = _sigmoid(kp[:, :, 2])
                kpts_all.append(kp)
        anchor_offset += H * W

    if not boxes_all:
        return []
    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    kpts = (np.concatenate(kpts_all, axis=0) if kpts_all
            else np.zeros((len(boxes), num_keypoints, 3), np.float32))

    dets = []
    for i in _nms(boxes, scores, iou_thres):
        box, kp = _map_back(boxes[i], kpts[i], letterbox)
        dets.append({"box": box, "score": float(scores[i]), "keypoints": kp})
    return dets


def decode(outputs, letterbox, num_keypoints=NUM_KEYPOINTS,
           num_classes=NUM_CLASSES, conf_thres=0.5, iou_thres=0.45):
    """Decode model outputs; auto-selects standard vs split-branch format."""
    std_ch = 4 + num_classes + num_keypoints * 3
    for o in outputs:
        sq = np.squeeze(np.asarray(o))
        if sq.ndim == 2 and std_ch in sq.shape:
            return _decode_standard(o, letterbox, num_keypoints, num_classes,
                                    conf_thres, iou_thres)
    return _decode_split(outputs, letterbox, num_keypoints, num_classes,
                         conf_thres, iou_thres)
