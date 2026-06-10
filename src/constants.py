"""Shared constants for the hand keypoint detector.

The model is a YOLO11-pose network trained on the Ultralytics
``hand-keypoints`` dataset: a single class (``hand``) with 21 keypoints per
hand laid out in the MediaPipe topology.
"""

# --- Keypoint layout (MediaPipe hand, 21 points) ---------------------------
NUM_KEYPOINTS = 21

KEYPOINT_NAMES = [
    "wrist",                                  # 0
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",          # 1-4
    "index_mcp", "index_pip", "index_dip", "index_tip",         # 5-8
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",     # 9-12
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",             # 13-16
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",         # 17-20
]

# Bones to draw between keypoints.
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm base
]

# Fingertip indices (handy for finger-counting / gesture logic later).
FINGERTIPS = [4, 8, 12, 16, 20]

# --- Model I/O -------------------------------------------------------------
# We follow the Rockchip rknn_model_zoo "yolov8_pose" export (airockchip fork):
# the detection head is split into raw branches with the DFL/decode removed
# (those ops quantise poorly / are slow on the NPU) and done on CPU instead.
#
# The exported model therefore yields 4 outputs:
#   * 3 detection branches, one per stride (8/16/32), each (1, DET_CH, H, W)
#         DET_CH = 4*REG_MAX + NUM_CLASSES   (box DFL logits + class logits)
#   * 1 keypoint branch (1, KPT_CH, total_anchors)
#         KPT_CH = NUM_KEYPOINTS * 3         (x, y, visibility), already decoded
INPUT_SIZE = 224                # MUST match the size baked into the .rknn model.
                                # Current hand_pose.rknn was exported at 224
                                # (-> 1029 anchors). To use 640, re-export the
                                # model at imgsz=640 and reconvert, then bump this.
NUM_CLASSES = 1                 # single class: "hand"
REG_MAX = 16                    # DFL bins per box side
STRIDES = (8, 16, 32)           # detection branch strides

DET_CHANNELS = 4 * REG_MAX + NUM_CLASSES   # 65
KPT_CHANNELS = NUM_KEYPOINTS * 3           # 63

# RKNN conversion target for ROCK 4D (RK3576 SoC).
RKNN_TARGET_PLATFORM = "rk3576"
