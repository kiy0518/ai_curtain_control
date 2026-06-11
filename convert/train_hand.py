#!/usr/bin/env python3
"""Train a YOLO11 hand-keypoints pose model — and FIX the left/right bias.

Run on a GPU machine or Google Colab. Ultralytics ships a ready-made
``hand-keypoints`` dataset (~26k images, 21 keypoints, 1 class, BOTH hands) that
downloads automatically — no manual image collection needed.

    pip install ultralytics
    python train_hand.py --model yolo11n-pose.pt --epochs 100 --imgsz 640 --export

Why the previous model only worked on ONE hand: a hand and its mirror are the
two chiralities (left/right). A half-trained model converges on one chirality
first and fails on the other. The fix is **train to convergence with L-R flip
augmentation on** (``--fliplr 0.5``) so both hands are learned equally. ``flip_idx``
is already defined in ``hand-keypoints.yaml`` (a hand has no internal L/R pairs,
so flipping keeps the same keypoint indices) — fliplr is safe and is the cure.

Free Colab disconnects on long runs. Mitigate:
  * ``--project /content/drive/MyDrive/ai_curtain_train`` so weights live on Drive
  * if it drops, rerun with ``--resume <project>/<name>/weights/last.pt``
  * ``--patience 20`` early-stops once val plateaus (saves hours)

The best weights land in ``<project>/<name>/weights/best.pt``. With ``--export``
they are also exported to ONNX (opset 12) — feed that to ``convert_rknn.py`` /
``Colab_hand_to_rknn.ipynb`` to get the ``.rknn`` for the board.
"""

import argparse


def main():
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n-pose.pt",
                   help="Base pose model to fine-tune")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Must match the board profile (hand_near = 640)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="hand_pose")
    p.add_argument("--project", default=None,
                   help="Output dir. On free Colab set a Google Drive path so a "
                        "disconnect does not lose progress.")
    p.add_argument("--fliplr", type=float, default=0.5,
                   help="L-R flip augmentation prob. KEEP > 0 — this is what "
                        "teaches BOTH hands. 0 reproduces the one-hand bug.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop patience (0 = off)")
    p.add_argument("--resume", default=None,
                   help="Path to a last.pt to resume an interrupted run")
    p.add_argument("--export", action="store_true",
                   help="After training, export best.pt to ONNX (opset 12)")
    args = p.parse_args()

    if args.resume:
        model = YOLO(args.resume)              # resumes with the saved train args
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data="hand-keypoints.yaml",        # auto-downloaded by Ultralytics
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            project=args.project,
            fliplr=args.fliplr,                # <-- the chirality fix
            patience=args.patience,
        )

    save_dir = model.trainer.save_dir
    best = save_dir / "weights" / "best.pt"
    print("Done. best weights:", best)

    if args.export:
        # Standard Ultralytics ONNX (opset 12). postprocess.py on the board
        # auto-detects this layout; convert_rknn.py builds the FP16 .rknn.
        onnx = YOLO(str(best)).export(format="onnx", opset=12, imgsz=args.imgsz)
        print("ONNX (opset 12):", onnx)


if __name__ == "__main__":
    main()
