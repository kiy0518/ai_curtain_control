# AI Curtain Control — 손 제스처로 커튼 제어 (ROCK 4D / RK3576 NPU)

Radxa ROCK 4D(RK3576)의 NPU에서 YOLOv8-pose 기반 **손 키포인트(21점)** 를 실시간 탐지하고,
손 제스처(✋ 열림 / ✊ 닫힘 / ✌️ 정지)를 인식해 화면 중앙 상단에 표시합니다.
IMX415 CSI 카메라 → NPU 추론(`rknnlite`) → 키포인트/제스처 → 웹 MJPEG 스트리밍.

```
┌── x86 PC / Colab ──────────────┐      ┌── ROCK 4D (this board) ───────────┐
│ Ultralytics 손 키포인트 학습    │      │ IMX415 CSI ─► 전처리(letterbox)   │
│   └► ONNX ─► hand_pose.rknn ───┼─scp─►│   └► RKNN NPU 추론 ─► 디코드/NMS  │
│        (convert/)              │      │        └► 스켈레톤 그리기 ─► 표시 │
└────────────────────────────────┘      └────────────────────────────────────┘
```

## 폴더 구조

```
ai_curtain_control/
├── run.py                  # 진입점: 카메라/이미지/동영상 → 추론 → 시각화
├── src/
│   ├── constants.py        # 21 키포인트 레이아웃, 스켈레톤, 입력 크기
│   ├── camera.py           # 프레임 소스 + letterbox 전처리
│   ├── hand_pose.py        # RKNNLite 래퍼 (NPU 추론)
│   ├── postprocess.py      # pose 출력 디코드 + NMS + 좌표 복원
│   └── draw.py             # 박스/스켈레톤/키포인트 렌더링
├── convert/                # (x86/Colab) 학습 + RKNN 변환 — convert/README.md
├── models/                 # hand_pose.rknn 을 여기에 둠
└── requirements-board.txt  # 보드용 의존성 (opencv)
```

## 시작하기

### 1. 카메라 활성화 (완료됨 → 재부팅 필요)
IMX415 오버레이(`rock-4d-radxa-camera-4k.dtbo`)는 이미 활성화했습니다. 적용하려면:
```bash
sudo reboot
```
재부팅 후 인식 확인:
```bash
ls /dev/video*            # 새 캡처 노드 확인
ls /dev/media*            # /dev/media0
dmesg | grep -i imx415    # 센서 probe 로그
```

### 2. 의존성 설치 (보드)
```bash
pip3 install -r requirements-board.txt
```

### 3. 모델 준비 (Colab · 학습 불필요)
손 21키포인트 **사전학습 모델**이 있으므로 학습이 필요 없습니다.
[convert/Colab_hand_to_rknn.ipynb](convert/Colab_hand_to_rknn.ipynb) 를 Colab에서 실행 → `hand_pose.rknn` 생성 → `models/` 로 복사. 자세히는 [convert/README.md](convert/README.md).

### 4-A. 웹 브라우저로 보기 (권장 · 디스플레이 불필요)
표준 라이브러리 MJPEG 서버. 같은 네트워크의 어떤 브라우저에서도 접속 가능.
```bash
# 카메라만 (모델 없이도 동작)
python3 serve.py

# 손 키포인트 오버레이까지 (모델 준비 후)
python3 serve.py --model models/hand_pose.rknn

# 해상도/포트 변경
python3 serve.py --width 1280 --height 720 --port 8080
```
브라우저에서  **http://<board-ip>:8080**  접속 (예: http://192.168.10.196:8080).
경로: `/`(뷰어), `/stream.mjpg`(MJPEG), `/snapshot.jpg`(단일 프레임).

### 4-B. 로컬 창 / 파일 처리
```bash
# 라이브 CSI 카메라 (로컬 디스플레이)
python3 run.py --model models/hand_pose.rknn --source "$(python3 -c 'import sys;sys.path.insert(0,"src");from camera import isp_gst_pipeline;print(isp_gst_pipeline())')"

# 단일 이미지 (카메라 없이 테스트)
python3 run.py --model models/hand_pose.rknn --source test.jpg --save out.jpg

# 헤드리스 → 동영상 저장
python3 run.py --model models/hand_pose.rknn --gst "$(python3 -c 'import sys;sys.path.insert(0,"src");from camera import isp_gst_pipeline;print(isp_gst_pipeline())')" --save out.mp4 --no-show
```

## 진행 상태
- [x] IMX415 카메라 오버레이 활성화 (`rock-4d-radxa-camera-4k.dtbo` + `u-boot-update`)
- [x] 카메라 인식·캡처 확정: **`/dev/video-camera0`(=video11)**, NV12, 1080p@60, ISP 자동구성
- [x] 보드 OpenCV(4.6.0, GStreamer 지원) 설치 + 캡처 검증
- [x] 웹 MJPEG 스트리밍 (`serve.py`) — 브라우저 라이브 확인 완료
- [x] 프로젝트 구조 / 추론 코드 골격
- [ ] `hand_pose.rknn` 생성 (Colab) — `convert/` → `models/` 로 복사
- [ ] 모델 적용 후 손 키포인트 오버레이 엔드투엔드 검증

## 저지연 스트리밍 설계
[humiro_fire_suppression](https://github.com/kiy0518/humiro_fire_suppression) 의 스트리밍 구조를 참고했습니다.
핵심은 **캡처 / 처리 / 서빙 스레드 분리 + 오래된 프레임 폐기(drop-old)** 로 지연 누적을 막는 것:
```
CameraThread ──raw_slot──▶ ProcessThread ──jpeg_slot──▶ HTTP 클라이언트
   (최신만 유지)            (NPU추론+그리기+JPEG)        (최신만 전송)
```
- `src/streaming.py: LatestSlot` = 그 프로젝트의 `ThreadSafeQueue(drop-oldest)` 를 size 1 로 특화한 것.
- 느린 클라이언트가 캡처를 막지 않고, 항상 가장 최신 프레임만 인코딩/전송 → 체감 지연 최소.
- 실측: 720p MJPEG **~59 fps**.

## 메모
- **NPU 코어**: RK3576은 NPU 코어가 여러 개입니다. `hand_pose.py` 에서 `NPU_CORE_AUTO` 사용 중이며, 멀티코어로 묶고 싶으면 `NPU_CORE_0_1_2` 등으로 변경.
- **양자화 정확도**: INT8에서 키포인트 정밀도가 떨어지면 변환을 `--no-quant`(FP16)로 비교.
- **카메라**: IMX415 → ISP 메인경로 `/dev/video-camera0`, NV12. 자동 노출(AE)이 수렴하는 데 첫 몇 프레임이 필요해 시작 직후엔 어둡습니다(정상).
```
