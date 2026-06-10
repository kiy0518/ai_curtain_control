"""Pose engine: owns the active model + profile, runs inference + gesture +
overlay, and supports **runtime model/profile hot-swap** (for dashboard model
selection). Thread-safe: the HTTP thread may call ``set_profile`` while the
process thread calls ``process``.
"""

import threading
import time

from profiles import get_profile, PROFILES
from hand_pose import HandPose
from gesture import GestureStabilizer
from draw import draw_detections, draw_gesture_banner


class PoseEngine:
    def __init__(self, profile_name, conf=0.3, controller=None, hold=3):
        self.conf = conf
        self.controller = controller
        self.gesture_enabled = True
        self.hold = max(1, int(hold))          # consecutive frames to confirm
        self.stabilizer = GestureStabilizer(hold=self.hold)

        self._lock = threading.Lock()
        self.model = None
        self.profile = None
        self._committed = None
        self.stats = {"infer_ms": 0.0, "count": 0, "raw": None, "gesture": None}
        self.set_profile(profile_name)

    # --- model lifecycle ---------------------------------------------------
    def set_profile(self, name, model_path=None):
        """Load ``name``'s model and atomically swap it in. Slow (~0.5s);
        called from the HTTP thread. Raises on unknown/failed model."""
        prof = get_profile(name)
        new = HandPose.from_profile(prof, conf_thres=self.conf,
                                    model_path=model_path)
        with self._lock:
            old, self.model, self.profile = self.model, new, prof
            self.stabilizer = GestureStabilizer(hold=self.hold)
            self._committed = None
        if old is not None:
            old.release()
        return prof

    def set_conf(self, conf):
        with self._lock:
            self.conf = conf
            if self.model is not None:
                self.model.conf_thres = conf

    def set_gesture_enabled(self, enabled):
        self.gesture_enabled = bool(enabled)

    def set_hold(self, hold):
        """Set the consecutive-frame count needed to confirm a gesture."""
        with self._lock:
            self.hold = max(1, int(hold))
            self.stabilizer = GestureStabilizer(hold=self.hold)
            self._committed = None

    def release(self):
        with self._lock:
            if self.model is not None:
                self.model.release()
                self.model = None

    # --- per-frame ---------------------------------------------------------
    def process(self, frame):
        # Infer INSIDE the lock so a concurrent set_profile() can't release the
        # model mid-inference (use-after-free → native crash). The swap waits
        # for the current frame (~tens of ms); model *loading* stays off-lock.
        with self._lock:
            if self.model is None:
                return frame
            prof = self.profile
            t = time.time()
            dets = self.model.infer(frame)
            infer_ms = (time.time() - t) * 1000.0

        draw_detections(frame, dets, skeleton=prof.skeleton,
                        highlight=prof.highlight, label=prof.name)

        raw = None
        if dets and self.gesture_enabled:
            best = max(dets, key=lambda d: d["score"])
            raw = prof.classify(best["keypoints"])

        committed = self.stabilizer.update(raw)
        # fire a curtain command only when the committed gesture changes
        if committed and committed != self._committed and self.controller:
            self.controller.command(committed, "gesture")
        self._committed = committed
        draw_gesture_banner(frame, committed)

        self.stats = {"infer_ms": round(infer_ms, 1), "count": len(dets),
                      "raw": raw, "gesture": committed}
        return frame

    # --- introspection for the dashboard ----------------------------------
    def state(self):
        with self._lock:
            prof = self.profile
        return {
            "profile": prof.name,
            "profile_desc": prof.desc,
            "imgsz": prof.imgsz,
            "num_keypoints": prof.num_keypoints,
            "conf": self.conf,
            "gesture_enabled": self.gesture_enabled,
            "hold": self.hold,
            **self.stats,
        }

    @staticmethod
    def available_profiles():
        return [{"name": p.name, "desc": p.desc, "imgsz": p.imgsz,
                 "num_keypoints": p.num_keypoints} for p in PROFILES.values()]
