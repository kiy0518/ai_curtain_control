#!/usr/bin/env python3
"""Train (or download) a YOLO11 hand-keypoints pose model.

Run on a GPU machine or Google Colab. Ultralytics ships a ready-made
``hand-keypoints`` dataset (21 keypoints, 1 class) that downloads automatically.

    pip install ultralytics
    python train_hand.py --model yolo11n-pose.pt --epochs 100 --imgsz 640

The best weights land in ``runs/pose/<name>/weights/best.pt`` — feed that to
``convert_rknn.py``. For a quick start you can skip training and convert the
COCO-pose pretrained model, but it detects *body* keypoints, not hands; train
on hand-keypoints for actual finger detection.
"""

import argparse


def main():
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n-pose.pt",
                   help="Base pose model to fine-tune")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="hand_pose")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data="hand-keypoints.yaml",   # auto-downloaded by Ultralytics
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )
    print("Done. Best weights: runs/pose/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
