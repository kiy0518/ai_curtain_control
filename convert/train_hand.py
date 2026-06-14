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
import time


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
    p.add_argument("--scale", type=float, default=0.9,
                   help="Scale aug gain: each image randomly scaled in "
                        "[1-scale, 1+scale] = [0.1, 1.9]. 0.9 teaches FAR(small) "
                        "AND near(large) hands -> extends detection range. "
                        "원본/큰 손도 그대로 학습됨(작은 손을 추가할 뿐).")
    p.add_argument("--mosaic", type=float, default=1.0,
                   help="Mosaic aug prob (4 images tiled -> more small hands).")
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop patience (0 = off)")
    p.add_argument("--resume", default=None,
                   help="Path to a last.pt to resume an interrupted run")
    p.add_argument("--export", action="store_true",
                   help="After training, export best.pt to ONNX (opset 12)")
    args = p.parse_args()

    print("=== 손 키포인트 학습 시작 ===", flush=True)
    print("  (처음 몇 분은 데이터셋 자동 다운로드 — 잠시 대기)", flush=True)
    model = YOLO(args.resume if args.resume else args.model)

    # 진행바(tqdm)는 끄되 에폭마다 한 줄씩 출력 → 저로그 + 진행 확인
    t0 = time.time()

    def _hb(tr):
        el = int(time.time() - t0)
        print("[진행] epoch %d/%d  (경과 %d분 %d초)"
              % (tr.epoch + 1, tr.epochs, el // 60, el % 60), flush=True)
    model.add_callback("on_fit_epoch_end", _hb)

    if args.resume:
        model.train(resume=True)
    else:
        model.train(
            data="hand-keypoints.yaml",        # auto-downloaded by Ultralytics
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            project=args.project,
            fliplr=args.fliplr,                # <-- the chirality fix
            scale=args.scale,                  # <-- 먼/작은 손까지 학습(거리 확장)
            mosaic=args.mosaic,                # <-- 작은 손 추가 노출
            patience=args.patience,
            verbose=False,                     # 상세 로그 끔(진행은 위 콜백)
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
