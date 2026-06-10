# 모델 준비 (Google Colab / x86 PC 에서 실행)

> ⚠️ `rknn-toolkit2`(변환 도구)는 **x86-64 리눅스 전용**입니다. ARM 보드(ROCK 4D)에서는 변환이 안 되고, 변환된 `.rknn`만 `rknnlite`로 실행합니다.
> x86 PC가 없으면 **Google Colab(무료, x86 리눅스)** 에서 그대로 돌립니다.

## ⚠️ 왜 표준 export 가 아닌가
Ultralytics 의 네이티브 `yolo export format=rknn` 은 **detection 모델만 지원하고 pose 는 미지원**입니다
([공식 문서](https://docs.ultralytics.com/integrations/rockchip-rknn/)). 따라서 손 키포인트(pose)는
Rockchip 공식 방식인 **airockchip 포크 export + rknn_model_zoo 후처리**를 따릅니다.
이 포크는 헤드를 분리하고 DFL/디코드를 제거(NPU 양자화·속도 개선)하며, 디코드는 보드 CPU(`src/postprocess.py`)에서 합니다.

## ✅ 가장 쉬운 길: Colab 노트북 (학습 불필요)

손 21키포인트 **사전학습 모델이 이미 존재**합니다. [`Colab_hand_to_rknn.ipynb`](Colab_hand_to_rknn.ipynb) 를
Google Colab에 올리고 셀을 순서대로 실행하면 끝입니다:

```
[Colab] best.pt (사전학습)
   │  ① airockchip 포크 exporter.py  → 분리브랜치 best.onnx (3 det + 1 kpt)
   │  ② rknn-toolkit2                → hand_pose.rknn (rk3576)
   ▼
보드 models/ 로 복사
```

사용 모델: **RionDsilvaCS/yolo-hand-pose** (YOLOv8n-pose, Kaggle 26k 손 키포인트셋)
편의를 위해 이 폴더에 [`best.pt`](best.pt) 로 이미 받아 두었습니다.

## 스크립트로 변환 (노트북 대신 CLI, x86 리눅스)

```bash
# ① airockchip 포크로 분리브랜치 ONNX 생성
git clone https://github.com/airockchip/ultralytics_yolov8
cd ultralytics_yolov8 && pip install -r requirements.txt onnx onnxslim
cp ../best.pt .
sed -i 's#^model:.*#model: best.pt#; s#^task:.*#task: pose#' ultralytics/cfg/default.yaml
export PYTHONPATH=./ && python ./ultralytics/engine/exporter.py   # -> best.onnx
cd ..

# ② ONNX → RKNN (rk3576). 기본 FP16, INT8 은 --int8 --dataset
pip install rknn-toolkit2==2.3.0
python convert_rknn.py ultralytics_yolov8/best.onnx --target rk3576 --out hand_pose.rknn
```

- 기본 **FP16**: 보정 이미지 불필요, 정확도 안정적, rk3576 NPU 가속. 우선 권장.
- **INT8**(`--int8 --dataset calib_list.txt`): 더 빠름, 손 이미지 20~200장 필요. 정확도 떨어지면 FP16 사용.

## (선택) 직접 학습
정확도를 더 높이려면 [`train_hand.py`](train_hand.py) 로 재학습 후 그 `best.pt`를 위와 동일하게 export/변환하세요.

## 보드로 복사
```bash
scp hand_pose.rknn radxa@192.168.10.196:~/Documents/ai_curtain_control/models/
```

## 참고: rknnlite 버전 정합
보드 런타임은 `rknnlite 2.3.0` / `librknnrt.so` 입니다. 변환 시 `rknn-toolkit2`도 **2.3.x** 로 맞추세요(주요 버전이 다르면 로드 실패 가능).
