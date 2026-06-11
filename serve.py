#!/usr/bin/env python3
"""Low-latency MJPEG web stream of the CSI camera, with optional hand-keypoint
overlay.

Architecture follows the reference humiro_fire_suppression project: separate
capture / processing / serving threads joined by drop-old single-slot buffers,
so a slow client never stalls capture and viewers always get the freshest
frame. Pure standard library on the HTTP side — no Flask.

Open ``http://<board-ip>:8080`` in any browser on the same network.

Examples:
    python3 serve.py                                   # camera only
    python3 serve.py --model models/hand_pose.rknn     # with keypoint overlay
    python3 serve.py --width 1280 --height 720 --fps 30 --quality 75
"""

import argparse
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")

from camera import isp_gst_pipeline                       # noqa: E402
from draw import (draw_detections, draw_fps, draw_gesture_banner,  # noqa: E402
                  draw_motion_debug)
from streaming import CameraThread, ProcessThread         # noqa: E402
from gesture import GestureStabilizer, extended_fingers     # noqa: E402
from profiles import get_profile                            # noqa: E402
import cv2                                                  # noqa: E402


INDEX_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yolo Finger Detect — Live</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
  h1{font-size:17px;padding:10px;margin:0;background:#1b1b1b;color:#4CAF50}
  img{max-width:100%;height:auto;border:1px solid #333}
  .info{font-size:12px;color:#888;padding:6px}
</style></head>
<body>
  <h1>ROCK 4D · IMX415 · 손 키포인트 라이브</h1>
  <img src="/stream.mjpg" alt="camera stream">
  <div class="info">/stream.mjpg · /snapshot.jpg</div>
</body></html>
"""


def make_handler(process_thread):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/snapshot.jpg":
                jpeg = process_thread.jpeg_slot.get()
                if jpeg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)

            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                last_ver = 0
                try:
                    while True:
                        # Block until a *newer* frame exists -> no busy loop,
                        # no stale frames, minimal latency.
                        jpeg, last_ver = process_thread.jpeg_slot.wait_newer(
                            last_ver, timeout=2.0)
                        if jpeg is None:
                            continue
                        chunk = (b"--FRAME\r\n"
                                 b"Content-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode()
                                 + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def parse_args():
    p = argparse.ArgumentParser(description="Low-latency web MJPEG stream")
    p.add_argument("--profile", default="hand_near",
                   help="model profile: hand_near (근거리 손) | body_far (원거리 팔포즈)"
                        " | body_motion (원거리 손목 움직임)")
    p.add_argument("--model", default=None, help="override .rknn path (default: profile's)")
    p.add_argument("--no-model", action="store_true", help="camera only, no inference")
    p.add_argument("--device", default=None, help="Camera device (default ISP node)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--quality", type=int, default=75, help="JPEG quality 1-100")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=None,
                   help="override input size — must match the .rknn (default: profile's)")
    p.add_argument("--debug", action="store_true",
                   help="Overlay gesture debug (per-finger states for hand profile)")
    return p.parse_args()


def main():
    args = parse_args()

    profile = get_profile(args.profile)
    model = None
    if not args.no_model:
        model_path = args.model or profile.model_path
        if not model_path.endswith(".rknn"):
            raise SystemExit("이 보드에서는 .rknn 모델만 사용하세요 (torch CPU 추론 SIGILL).")
        print(f"Profile: {profile.name} ({profile.desc})")
        print(f"Loading model: {model_path} (imgsz={args.imgsz or profile.imgsz})")
        from hand_pose import HandPose            # NPU (rknnlite)
        model = HandPose.from_profile(profile, conf_thres=args.conf,
                                      model_path=args.model, imgsz=args.imgsz)

    if args.device:
        pipeline = isp_gst_pipeline(args.device, args.width, args.height, args.fps)
    else:
        pipeline = isp_gst_pipeline(width=args.width, height=args.height, fps=args.fps)

    # Gesture recognition: classify the most confident detection each frame
    # (using the active profile's classifier), debounce, draw Korean banner.
    stabilizer = GestureStabilizer(hold=5)
    tracker = profile.make_classifier() if profile.make_classifier else None
    last_event = {"label": None, "ts": 0.0}   # 이벤트형: 배너 유지 표시용
    is_hand = profile.num_keypoints == 21

    def draw_with_gesture(frame, dets):
        draw_detections(frame, dets, skeleton=profile.skeleton,
                        highlight=profile.highlight, label=profile.name)
        if tracker is not None:               # 이벤트형(손목 움직임) 분류기
            now = time.time()
            evt = tracker.update(dets, now)
            if evt:
                last_event["label"], last_event["ts"] = evt, now
            if args.debug:
                draw_motion_debug(frame, tracker)
            shown = (last_event["label"]
                     if now - last_event["ts"] < 2.0 else None)
            draw_gesture_banner(frame, shown)
            return
        label = None
        if dets:
            best = max(dets, key=lambda d: d["score"])
            label = profile.classify(best["keypoints"])
            if args.debug:
                if is_hand:
                    st = extended_fingers(best["keypoints"])
                    txt = "I:%d M:%d R:%d P:%d n=%d raw=%s" % (
                        st["index"], st["middle"], st["ring"], st["pinky"],
                        sum(st.values()), label)
                else:
                    txt = "raw=%s" % label
                cv2.putText(frame, txt, (8, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                            cv2.LINE_AA)
        draw_gesture_banner(frame, stabilizer.update(label))

    cam = CameraThread(pipeline)
    proc = ProcessThread(
        cam.slot, model=model, jpeg_quality=args.quality,
        draw_fn=draw_with_gesture if model else None,
        fps_fn=draw_fps,
    )
    cam.start()
    proc.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(proc))
    print(f"\n  ▶ 스트리밍 시작 →  http://<board-ip>:{args.port}\n"
          f"    (이 보드: http://localhost:{args.port})   Ctrl+C 종료\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        proc.stop()
        cam.stop()
        server.shutdown()
        if model is not None:
            model.release()


if __name__ == "__main__":
    main()