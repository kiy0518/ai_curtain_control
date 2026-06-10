#!/usr/bin/env python3
"""Convert an airockchip-exported YOLOv8-pose ONNX to RKNN for the RK3576 NPU.

Run on **x86-64 Linux** (Google Colab works) — ``rknn-toolkit2`` does not run on
the ARM board. The board only runs the resulting ``.rknn`` via ``rknnlite``.

IMPORTANT — pose models are NOT supported by Ultralytics' native ``format=rknn``
export. You must first produce an RKNN-optimised ONNX with the airockchip fork
(which splits the head and removes DFL/decode):

    git clone https://github.com/airockchip/ultralytics_yolov8
    cd ultralytics_yolov8 && pip install -r requirements.txt
    # edit ultralytics/cfg/default.yaml ->  model: best.pt   task: pose
    export PYTHONPATH=./ && python ./ultralytics/engine/exporter.py   # -> best.onnx

Then convert that ONNX here (config matches rknn_model_zoo/yolov8_pose):

    python convert_rknn.py best.onnx --target rk3576 --out hand_pose.rknn
    # INT8 (faster, needs calibration images):
    python convert_rknn.py best.onnx --target rk3576 --dataset calib_list.txt --int8

The Colab notebook ``Colab_hand_to_rknn.ipynb`` does all of the above for you.
"""

import argparse
import os


def convert(onnx_path, out_path, target, calib_list, quantize):
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    # YOLO inputs are RGB uint8 normalised by /255 -> mean=0, std=255.
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform=target)

    if rknn.load_onnx(model=onnx_path) != 0:
        raise RuntimeError("load_onnx failed")

    do_quant = quantize and calib_list is not None and os.path.exists(calib_list)
    if quantize and not do_quant:
        print("[warn] no calibration list -> building FP16 model (do_quant=False)")
    if rknn.build(do_quantization=do_quant,
                  dataset=calib_list if do_quant else None) != 0:
        raise RuntimeError("build failed")

    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError("export_rknn failed")
    print(f"[rknn] exported -> {out_path}  (quantized={do_quant}, target={target})")
    rknn.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("onnx", help="airockchip-exported pose ONNX (split-branch)")
    p.add_argument("--target", default="rk3576")
    p.add_argument("--out", default="hand_pose.rknn")
    p.add_argument("--dataset", default=None,
                   help="Text file listing calibration images (INT8 only)")
    p.add_argument("--int8", action="store_true",
                   help="INT8 quantization (default is FP16)")
    args = p.parse_args()

    convert(args.onnx, args.out, args.target, args.dataset, quantize=args.int8)


if __name__ == "__main__":
    main()
