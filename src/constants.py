"""Shared constants and keypoint layouts.

Two keypoint topologies are supported (see ``profiles.py`` for model profiles):

* **hand** — 21 keypoints (MediaPipe hand), close-range finger gestures.
* **body** — 17 keypoints (COCO pose), far-range arm gestures.

The exported RKNN models follow the Rockchip rknn_model_zoo "yolov8_pose"
format. Channel counts depend on the keypoint count, so ``postprocess.decode``
is parameterised by ``num_keypoints``/``num_classes`` (these globals are only
defaults for the hand model / backward compatibility).
"""

# --- Hand layout (MediaPipe, 21 points) ------------------------------------
NUM_KEYPOINTS = 21              # default (hand); profiles override
KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIPS = [4, 8, 12, 16, 20]

# --- Body layout (COCO pose, 17 points) ------------------------------------
# 0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear, 5 L-shoulder, 6 R-shoulder,
# 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip,
# 13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle
BODY_KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]
BODY_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 6), (5, 11), (6, 12), (11, 12),     # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
    (0, 5), (0, 6),                         # head to shoulders
]

# --- Gesture labels (shared by hand & body classifiers) --------------------
GESTURE_KR = {"OPEN": "열림", "CLOSE": "닫힘", "STOP": "정지"}

# --- Model I/O defaults ----------------------------------------------------
INPUT_SIZE = 224                # default; each profile sets its own imgsz
NUM_CLASSES = 1                 # 1 class (hand) or (person) — both single-class
REG_MAX = 16                    # DFL bins per box side
STRIDES = (8, 16, 32)           # detection branch strides

DET_CHANNELS = 4 * REG_MAX + NUM_CLASSES   # 65 (single class)
KPT_CHANNELS = NUM_KEYPOINTS * 3           # 63 (hand)

RKNN_TARGET_PLATFORM = "rk3576"
