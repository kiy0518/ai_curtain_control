#!/usr/bin/env python3
"""Hand keypoint detection on the ROCK 4D NPU.

Examples:
    # Live CSI camera (after enabling the overlay + reboot)
    python3 run.py --model models/hand_pose.rknn --source /dev/video0

    # Rockchip ISP via GStreamer
    python3 run.py --model models/hand_pose.rknn \
        --gst "v4l2src device=/dev/video11 ! video/x-raw,format=NV12,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink"

    # Single image (no camera needed)
    python3 run.py --model models/hand_pose.rknn --source test.jpg --save out.jpg

    # Headless: write annotated video instead of displaying a window
    python3 run.py --model models/hand_pose.rknn --source /dev/video0 --save out.mp4 --no-show
"""

import argparse
import sys
import time

import cv2

sys.path.insert(0, "src")

from camera import open_source, is_image_file       # noqa: E402
from hand_pose import HandPose                       # noqa: E402
from draw import draw_detections, draw_fps           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="YOLO hand keypoint detection (RK3576 NPU)")
    p.add_argument("--model", required=True, help="Path to .rknn model")
    p.add_argument("--source", default="/dev/video0",
                   help="Camera index, /dev/videoN, or image/video file")
    p.add_argument("--gst", default=None,
                   help="Explicit GStreamer pipeline (overrides --source)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--save", default=None, help="Write annotated output to this path")
    p.add_argument("--no-show", action="store_true", help="Do not open a display window")
    return p.parse_args()


def run_image(model, path, save, show):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")
    dets = model.infer(frame)
    draw_detections(frame, dets)
    print(f"Detected {len(dets)} hand(s)")
    if save:
        cv2.imwrite(save, frame)
        print(f"Saved -> {save}")
    if show:
        cv2.imshow("hand-pose", frame)
        cv2.waitKey(0)


def run_stream(model, source, save, show):
    cap = open_source(source)
    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        writer = cv2.VideoWriter(save, fourcc, 30.0, (w, h))

    fps, last = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended / read failed.")
                break

            dets = model.infer(frame)
            draw_detections(frame, dets)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            draw_fps(frame, fps)

            if writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("hand-pose", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    args = parse_args()
    source = args.gst if args.gst else args.source
    show = not args.no_show

    with HandPose(args.model, conf_thres=args.conf, iou_thres=args.iou) as model:
        if not args.gst and is_image_file(source):
            run_image(model, source, args.save, show)
        else:
            run_stream(model, source, args.save, show)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
