"""Pose engine: owns the active model + profile, runs inference + gesture +
overlay, and supports **runtime model/profile hot-swap** (for dashboard model
selection). Thread-safe: the HTTP thread may call ``set_profile`` while the
process thread calls ``process``.
"""

import threading
import time

import cv2

from profiles import get_profile, PROFILES
from hand_pose import HandPose
from gesture import GestureStabilizer
import gesture_motion
from draw import draw_detections, draw_hud, draw_motion_debug

EVENT_HUD_SEC = 2.0    # 이벤트형 제스처를 HUD에 유지 표시하는 시간


class PoseEngine:
    def __init__(self, profile_name, conf=0.3, controller=None, hold=3, flip=False,
                 motion_hold_sec=None, motion_refractory_sec=None):
        self.conf = conf
        self.controller = controller
        self.gesture_enabled = True
        self.flip = bool(flip)                 # mirror video L-R (text stays normal)
        self.hold = max(1, int(hold))          # consecutive frames to confirm
        # body_motion 전용 타이밍 (정지 유지 / 명령 후 불응) — 런타임 조정 가능
        self.motion_hold_sec = (float(motion_hold_sec) if motion_hold_sec
                                else gesture_motion.HOLD_SEC)
        self.motion_refractory_sec = (float(motion_refractory_sec)
                                      if motion_refractory_sec
                                      else gesture_motion.REFRACTORY_SEC)
        self.stabilizer = GestureStabilizer(hold=self.hold)

        self._lock = threading.Lock()
        self.model = None
        self.profile = None
        self.tracker = None                    # 이벤트형 분류기 (프로파일에 따라)
        self._committed = None
        self._last_event = None                # 이벤트형: 마지막 확정 제스처
        self._last_event_ts = 0.0
        self.gesture_event = 0                  # 확정 발행마다 +1 (클라이언트 비프용)
        self.gesture_event_label = None
        self.stats = {"infer_ms": 0.0, "count": 0, "raw": None, "gesture": None}
        self.set_profile(profile_name)

    # --- model lifecycle ---------------------------------------------------
    def set_profile(self, name, model_path=None):
        """Load ``name``'s model and atomically swap it in. Slow (~0.5s);
        called from the HTTP thread. Raises on unknown/failed model."""
        prof = get_profile(name)
        new = HandPose.from_profile(prof, conf_thres=self.conf,
                                    model_path=model_path)
        tracker = prof.make_classifier() if prof.make_classifier else None
        if tracker is not None and hasattr(tracker, "set_timing"):
            tracker.set_timing(self.motion_hold_sec, self.motion_refractory_sec)
        with self._lock:
            old, self.model, self.profile = self.model, new, prof
            self.tracker = tracker             # 상태 분류기는 스왑 시 새로 생성
            self.stabilizer = GestureStabilizer(hold=self.hold)
            self._committed = None
            self._last_event, self._last_event_ts = None, 0.0
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

    def set_flip(self, on):
        self.flip = bool(on)

    def set_motion_timing(self, hold_sec=None, refractory_sec=None):
        """body_motion의 정지 유지 / 불응 시간을 런타임에 변경(설정/스왑에 보존)."""
        with self._lock:
            if hold_sec is not None:
                self.motion_hold_sec = float(hold_sec)
            if refractory_sec is not None:
                self.motion_refractory_sec = float(refractory_sec)
            if self.tracker is not None and hasattr(self.tracker, "set_timing"):
                self.tracker.set_timing(self.motion_hold_sec,
                                        self.motion_refractory_sec)

    def _fire_event(self, label):
        self.gesture_event += 1
        self.gesture_event_label = label

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
            # Mirror BEFORE inference/drawing so keypoints/boxes/HUD align and
            # the text (drawn afterwards) stays readable (not mirrored).
            if self.flip:
                frame[:] = cv2.flip(frame, 1)
            prof = self.profile
            tracker = self.tracker
            t = time.time()
            dets = self.model.infer(frame)
            infer_ms = (time.time() - t) * 1000.0

        draw_detections(frame, dets, skeleton=prof.skeleton,
                        highlight=prof.highlight, label=prof.name)

        score = max((d["score"] for d in dets), default=0.0)

        if tracker is not None:
            # 이벤트형(움직임) 분류기: 확정 순간 1회 라벨 + 자체 디바운싱.
            # 미검출 프레임에서도 update를 불러야 소실 안전 STOP이 동작한다.
            now = time.time()
            raw = tracker.update(dets, now) if self.gesture_enabled else None
            if raw and self.controller:
                self.controller.command(raw, "gesture")
            if raw:
                self._last_event, self._last_event_ts = raw, now
                self._fire_event(raw)
            committed = (self._last_event
                         if now - self._last_event_ts < EVENT_HUD_SEC else None)
            draw_motion_debug(frame, tracker)
            draw_hud(frame, committed, None, 0, self.hold, score)
        else:
            raw = None
            if dets and self.gesture_enabled:
                best = max(dets, key=lambda d: d["score"])
                raw = prof.classify(best["keypoints"])
            committed = self.stabilizer.update(raw)
            # fire a curtain command only when the committed gesture changes
            if committed and committed != self._committed:
                if self.controller:
                    self.controller.command(committed, "gesture")
                self._fire_event(committed)
            self._committed = committed
            # top-centre HUD: gesture + confirm counter (N/hold) + confidence
            draw_hud(frame, committed, self.stabilizer.candidate,
                     self.stabilizer.count, self.hold, score)

        self.stats = {"infer_ms": round(infer_ms, 1), "count": len(dets),
                      "raw": raw, "gesture": committed, "score": round(score, 2)}
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
            "flip": self.flip,
            "motion_hold_sec": round(self.motion_hold_sec, 1),
            "motion_refractory_sec": round(self.motion_refractory_sec, 1),
            "event_seq": self.gesture_event,
            "event_label": self.gesture_event_label,
            **self.stats,
        }

    @staticmethod
    def available_profiles():
        return [{"name": p.name, "desc": p.desc, "imgsz": p.imgsz,
                 "num_keypoints": p.num_keypoints} for p in PROFILES.values()]
