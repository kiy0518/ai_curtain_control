"""Low-latency streaming primitives.

Design mirrors the reference C++ project (humiro_fire_suppression): capture,
processing, and network serving run on separate threads connected by
*drop-old* single-slot buffers. A slow HTTP client can never stall capture,
and stale frames are discarded so viewers always see the freshest image —
this is what keeps end-to-end latency low.

    CameraThread  --raw_slot-->  ProcessThread  --jpeg_slot-->  HTTP clients
      (grab)        (drop old)    (infer+draw+encode)  (drop old)
"""

import threading
import time

import cv2

from camera import open_source


class LatestSlot:
    """A thread-safe single-item slot that always keeps only the newest value.

    Equivalent to the reference project's ``ThreadSafeQueue`` with the
    drop-oldest policy, specialised to size 1 — the minimum-latency case.
    """

    def __init__(self):
        self._value = None
        self._version = 0
        self._cond = threading.Condition()

    def set(self, value):
        with self._cond:
            self._value = value
            self._version += 1
            self._cond.notify_all()

    def get(self):
        """Return the current value immediately (may be None)."""
        with self._cond:
            return self._value

    def wait_newer(self, last_version, timeout=1.0):
        """Block until a value newer than ``last_version`` arrives.

        Returns ``(value, version)``. On timeout returns the current value so
        callers can re-send the last frame (keeps MJPEG connections alive).
        """
        with self._cond:
            if self._version <= last_version:
                self._cond.wait(timeout)
            return self._value, self._version


class CameraThread(threading.Thread):
    """Continuously grab frames into a drop-old slot. Never blocks on consumers."""

    def __init__(self, pipeline):
        super().__init__(daemon=True)
        self.pipeline = pipeline
        self.slot = LatestSlot()
        self._running = True
        self.capture_fps = 0.0

    def run(self):
        cap = open_source(self.pipeline)
        last = time.time()
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                self.slot.set(frame)
                now = time.time()
                self.capture_fps = 0.9 * self.capture_fps + 0.1 / max(now - last, 1e-6)
                last = now
        finally:
            cap.release()

    def stop(self):
        self._running = False


class ProcessThread(threading.Thread):
    """Take the newest captured frame, optionally run inference + draw, encode
    to JPEG, publish into a drop-old slot. Skips frames it can't keep up with."""

    def __init__(self, camera_slot, model=None, jpeg_quality=80, draw_fn=None,
                 fps_fn=None, process_fn=None):
        super().__init__(daemon=True)
        self.camera_slot = camera_slot
        self.model = model
        self.jpeg_quality = jpeg_quality
        self.draw_fn = draw_fn
        self.fps_fn = fps_fn
        # process_fn(frame) does the whole infer+draw step (e.g. PoseEngine);
        # takes precedence over model/draw_fn when given.
        self.process_fn = process_fn

        self.jpeg_slot = LatestSlot()
        self._running = True
        self.process_fps = 0.0

    def run(self):
        last = time.time()
        last_ver = 0
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        while self._running:
            frame, last_ver = self.camera_slot.wait_newer(last_ver, timeout=1.0)
            if frame is None:
                continue

            if self.process_fn is not None:
                self.process_fn(frame)
            elif self.model is not None:
                dets = self.model.infer(frame)
                if self.draw_fn:
                    self.draw_fn(frame, dets)

            now = time.time()
            self.process_fps = 0.9 * self.process_fps + 0.1 / max(now - last, 1e-6)
            last = now
            if self.fps_fn:
                self.fps_fn(frame, self.process_fps)

            ok, buf = cv2.imencode(".jpg", frame, params)
            if ok:
                self.jpeg_slot.set(buf.tobytes())

    def stop(self):
        self._running = False