# AI Curtain Control — 손 제스처로 커튼 제어 (ROCK 4D / RK3576 NPU)

Radxa ROCK 4D(RK3576)의 NPU에서 **YOLOv8-pose 손 키포인트(21점)** 를 실시간 탐지하고,
손 제스처(✋ 열림 / ✊ 닫힘 / ✌️ 정지)를 인식해 화면 중앙 상단에 한글로 표시합니다.
IMX415 CSI 카메라 → NPU 추론(`rknnlite`) → 키포인트/제스처 → 웹 MJPEG 스트리밍.

```
┌── x86 / Google Colab ───────────┐        ┌── ROCK 4D (이 보드) ────────────────┐
│ YOLOv8n-pose best.pt            │        │ IMX415 CSI ─► ISP 캡처(GStreamer)   │
│  └► ONNX(opset12) ─► .rknn ─────┼─ 복사 ─►│  └► RKNN NPU 추론(rknnlite)         │
│     (rk3576, FP16)              │        │     └► 후처리 ─► 제스처 ─► 웹스트림  │
└─────────────────────────────────┘        └─────────────────────────────────────┘
```

---

## 1. 하드웨어 / 환경

| 항목 | 내용 |
|------|------|
| 보드 | Radxa ROCK 4D (Rockchip **RK3576**, 6 TOPS NPU) |
| OS | Debian 12 (bookworm), 커널 6.1.84 (rk2410) |
| 카메라 | **IMX415** CSI (Radxa Camera 4K), 1080p@60 |
| 캡처 노드 | `/dev/video-camera0` (= video11, `rkisp_mainpath`), NV12 |
| NPU 런타임 | `rknnlite` 2.3.0, `librknnrt.so` 2.3.0 |
| OpenCV | apt `python3-opencv` 4.6 (**GStreamer 지원 필수**) |

---

## 2. 폴더 구조

```
ai_curtain_control/
├── serve.py                # 웹 MJPEG 스트리밍 + 제스처 (메인 실행)
├── run.py                  # 로컬 창/이미지/동영상 실행
├── src/
│   ├── constants.py        # 21키포인트 레이아웃, INPUT_SIZE 등
│   ├── camera.py           # ISP GStreamer 캡처 + letterbox
│   ├── hand_pose.py        # RKNNLite NPU 추론 (.rknn)
│   ├── hand_pose_torch.py  # (참고) CPU/ultralytics 백엔드 — 이 보드선 torch SIGILL로 미사용
│   ├── postprocess.py      # 출력 디코드 (분리브랜치/표준 자동감지) + NMS
│   ├── gesture.py          # 손가락 펴짐 판정 + 제스처 분류 + 떨림방지
│   ├── draw.py             # 박스/스켈레톤/키포인트 + 한글 제스처 배너(PIL)
│   └── streaming.py        # 저지연 스레드(캡처/처리/서빙 분리, drop-old)
├── models/                 # hand_pose*.rknn (224 / 320 / 640)
├── convert/                # Colab 변환 노트북 + 스크립트 + best.pt
└── requirements-board.txt
```

---

## 3. 실행

### 대시보드 (메인 — `app.py`)
```bash
cd ~/Documents/ai_curtain_control
bash deploy/install.sh          # 최초 1회: opencv·cloudflared·config.env
python3 app.py                  # 프로파일/포트 등은 config.env 로 조정
python3 app.py --profile body_far   # 또는 플래그로 override
```
- 브라우저 **http://<board-ip>:8080** → **로그인**(기본 비밀번호 `admin`, 접속 후 변경).
- 기능: 라이브 영상, 커튼 상태/제어(placeholder), **모델 런타임 전환**, 스케줄(시간·일출/일몰),
  원격 접속 토글(cloudflared), 시스템 상태, 비밀번호 변경.
- 부팅 자동실행: `deploy/ai-curtain.service` (systemd).
- 운영: 로그 `logs/curtain.log`(회전), 헬스체크 `GET /healthz`(무인증).

### 부팅 자동 실행 (systemd)
```bash
sudo cp deploy/ai-curtain.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ai-curtain
journalctl -u ai-curtain -f
```

### (참고) 단순 스트리밍/디버그 — `serve.py`
인증/대시보드 없이 스트림만:
```bash
python3 serve.py --model models/hand_pose_640.rknn --imgsz 640 --conf 0.3 --debug
```
> ⚠️ `--imgsz` 는 `.rknn` 입력 크기와 일치해야 합니다(모델에 고정). app.py 는 프로파일이 자동 지정.

---

## 4. 제스처 인식

21개 키포인트로 각 손가락이 펴졌는지 판정 → 제스처 분류:

### 손(hand_near) 제스처
| 손 모양 | 인식 | 화면 표시 |
|---------|------|-----------|
| 👈 엄지만 펴서 **옆으로** | OPEN | **열림** |
| 👍 엄지만 펴서 **위로** | CLOSE | **닫힘** |
| 🖐 **손바닥 모두 펴기** | STOP | **정지** |

- 손바닥(검지·중지·약지·새끼 4개 펴짐) → 정지. 주먹+엄지 방향(위/옆) → 열림/닫힘. 그냥 주먹은 무시.
- **떨림 방지**: 같은 제스처 5프레임 연속 유지 시 표시 (`GestureStabilizer`).

### 전신(body_far) 제스처 — 원거리
| 자세 | 인식 |
|------|------|
| 오른팔 수평 | 열림 |
| 왼팔 수평 | 닫힘 |
| 양팔 X 교차 | 정지 |

- 한글 배너는 cv2가 한글을 못 그려서 **PIL + NotoSansCJK** 폰트로 렌더링.

---

## 5. 보드 초기 셋업 기록 (재현용)

처음 보드에서 한 작업 — 동일 환경 재구성 시 참고:

```bash
# (1) IMX415 카메라 오버레이 활성화  ※ rename 만으론 부족, u-boot-update 필수
sudo mv /boot/dtbo/rock-4d-radxa-camera-4k.dtbo.disabled \
        /boot/dtbo/rock-4d-radxa-camera-4k.dtbo
sudo u-boot-update          # extlinux.conf 에 fdtoverlays 반영
sudo reboot
# 재부팅 후: /dev/video-camera0, /dev/media0 생성, dmesg 에 "Detected imx415 id"

# (2) OpenCV (GStreamer 포함) — pip opencv 는 GStreamer 미포함이라 apt 사용
sudo apt-get install -y python3-opencv

# (3) 카메라 캡처 확인 (NV12, ISP 자동구성)
#   첫 몇 프레임은 자동노출(AE) 수렴 전이라 어두움 — 정상
```

> ⚠️ **torch CPU 추론은 이 보드에서 불가**: PyPI torch 휠이 Cortex-A53/A72 에서 `Illegal instruction(SIGILL)` 로 크래시. 그래서 추론은 **NPU(RKNN) 전용**입니다.

---

## 6. 모델 변환 (Google Colab) — ⭐ 최종 성공 절차

> `rknn-toolkit2`(변환 도구)는 **x86-64 Linux 전용**이라 ARM 보드에선 안 됨 → Colab(무료 x86) 사용.
> Colab 은 Python 3.12 + numpy2 환경이라 버전 충돌이 많아, 아래가 **실제로 성공한 조합**입니다.
> 노트북: [`convert/Colab_hand_to_rknn.ipynb`](convert/Colab_hand_to_rknn.ipynb)

### 6.1 모델
- 사전학습: **RionDsilvaCS/yolo-hand-pose** (YOLOv8n-pose, 손 21키포인트, Kaggle 26k셋)
- `wget https://raw.githubusercontent.com/RionDsilvaCS/yolo-hand-pose/main/model/best.pt`

### 6.2 셀 1 — ONNX export (표준 ultralytics, opset 고정)
```python
!wget -q -O best.pt https://raw.githubusercontent.com/RionDsilvaCS/yolo-hand-pose/main/model/best.pt
!pip install -q ultralytics onnx onnxslim
from ultralytics import YOLO
YOLO('best.pt').export(format='onnx', opset=12, simplify=True, imgsz=640)  # 224/320/640
```
- **`opset=12` 명시가 핵심**: 안 하면 `get_latest_opset()` → `vars(torch.onnx)` 가 깨진 onnx fx 모듈을 import 해서 `onnxscript.ParamSchema` 에러 발생.
- ❌ airockchip 포크 export 는 Colab 최신 torch 와 계속 충돌(onnxscript/torchvision) → **표준 export 채택**.
- (참고) torch↔torchvision 불일치 시 `pip install torchvision==0.19.0`.

### 6.3 셀 2-A — rknn-toolkit2 설치 (버전 고정) → **런타임 재시작**
```python
!wget -q -O req.txt https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit2/packages/x86_64/requirements_cp312-2.3.2.txt
!pip install -q -r req.txt rknn-toolkit2==2.3.2
!pip install -q --force-reinstall "onnx==1.18.0" "numpy==1.26.4" "scipy==1.13.1"
# → 런타임 → 세션 다시 시작 (numpy 다운그레이드 반영)
```
충돌 → 해결 매핑:
| 증상 | 원인 | 해결 |
|------|------|------|
| `No module named 'numpy.char'` | numpy2 로딩 상태에서 1.26 다운그레이드 | **런타임 재시작** |
| scipy import 깨짐 | numpy2용 scipy ↔ numpy1.26 | `scipy==1.13.1` 고정 |
| `name 'exit' is not defined` | Colab 커널에 `exit` 내장 없음 + req 누락 | `req.txt`로 `fast-histogram` 설치 + 6.4의 exit 보충 |
| `module 'onnx' has no attribute 'mapping'` | **onnx 1.19+ 에서 mapping 제거** | **`onnx==1.18.0`** 고정 |

### 6.4 셀 2-B — RKNN 변환 (재시작 후) + 다운로드
```python
import builtins, sys, os
builtins.exit = sys.exit            # Colab 커널의 빠진 exit 보충
from rknn.api import RKNN
r = RKNN(verbose=False)
r.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3576')
assert r.load_onnx(model='/content/best.onnx') == 0
assert r.build(do_quantization=False) == 0           # FP16 (NPU 가속, 보정이미지 불필요)
assert r.export_rknn('/content/hand_pose.rknn') == 0
r.release()
from google.colab import files; files.download('/content/hand_pose.rknn')
```
- 입력 RGB, `/255` 정규화 → `mean=0, std=255`. 타겟 `rk3576`.
- **FP16**(`do_quantization=False`) 채택: 보정 이미지 불필요, 정확도 안정적.
- 산출물은 **표준 디코드 출력** `(1, 68, N)` (N=anchor수; 224→1029, 640→8400). 보드 `postprocess.py` 가 자동 감지.

### 6.5 보드 적용
```bash
# 받은 파일을 보드 models/ 로 복사 (scp 또는 VSCode 드래그)
scp hand_pose_640.rknn radxa@<board-ip>:~/Documents/ai_curtain_control/models/
```
> 변환은 toolkit **2.3.2**로 했지만 보드 런타임 **2.3.0**에서 정상 로드됨(패치 차이 호환). 모델 로그에 `librknnrt version: 2.3.0` + `Model toolkit version: 2.3.2` 확인.

---

## 7. 저지연 웹 스트리밍 설계

[humiro_fire_suppression](https://github.com/kiy0518/humiro_fire_suppression) 의 구조 참고 —
**캡처 / 처리 / 서빙 스레드 분리 + 오래된 프레임 폐기(drop-old)** 로 지연 누적 방지:
```
CameraThread ──raw_slot──▶ ProcessThread ──jpeg_slot──▶ HTTP 클라이언트
  (최신만 유지)         (NPU추론+제스처+JPEG)        (최신만 전송)
```
- `src/streaming.py: LatestSlot` = drop-old 큐를 size 1 로 특화. 느린 클라이언트가 캡처를 막지 않음.

---

## 8. 성능 (실측, NPU FP16)

| 입력 | 추론 | 특징 |
|------|------|------|
| 224 | ~20 ms (≈50 fps) | 가장 빠름, 가까운 손 충분 |
| 320 | 중간 | 속도·정확도 균형 |
| 640 | 느림 | 작은/먼 손 유리 |

---

## 9. 미해결 / 개선 과제

- [ ] **✌️ 정지(STOP/브이) 인식 불안정** — 모델의 접힌 손가락 키포인트가 부정확. `--debug` 로 `n` 값 확인 후 판정식(손목-MCP 기준 등) 개선 필요.
- [ ] 사전학습 모델이 README상 **"50%만 학습"** → 정확도 한계. 재학습 또는 더 나은 가중치로 교체 검토.
- [ ] 입력 크기(224/320/640) 최종 선택 — 속도 vs 정확도.
- [ ] 인식된 제스처 → **실제 커튼 모터/릴레이 제어** 연동(GPIO 등) 미구현.
- [ ] 부팅 시 자동 실행(systemd 서비스) 미설정.

---

## 10. 의존성 (보드)
```bash
pip3 install -r requirements-board.txt   # opencv 는 apt python3-opencv 권장(GStreamer)
# rknnlite, numpy 는 시스템 기본 제공
```
