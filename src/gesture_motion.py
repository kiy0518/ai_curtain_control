"""Wrist-motion gestures from COCO 17-keypoint body pose (far-range control).

손모양(hand_near)은 ~2m 이내에서만 동작하므로, 원거리에서는 전신 모델의
**손목 키포인트 궤적**으로 판정한다. 좌표를 어깨너비로 정규화하므로 거리와
무관하게 같은 임계값이 적용된다(거리 불변).

  * OPEN  (열림): 손을 어깨 높이 이상으로 들고 우→좌 쓸기 (사용자 기준)
  * CLOSE (닫힘): 손을 어깨 높이 이상으로 들고 좌→우 쓸기
  * STOP  (정지): 손을 들고 HOLD_SEC 동안 거의 정지 — 손바닥을 들어 보이며
                  멈추는 동작. (원거리에선 손가락 모양 인식이 불가하므로
                  '들고 멈춤'으로 판정; 근거리 손바닥은 hand_near가 담당)

**이벤트형** 분류기: ``update()``는 제스처가 확정되는 순간 라벨을 한 번만
반환하고 REFRACTORY_SEC 동안 불응한다. 정적 분류기(gesture.py 등)의
hold-N-frames 안정화와 달리 순간 동작에 맞춘 디바운싱이다.

안전 기본값: 직전에 이 분류기가 OPEN/CLOSE를 발행한 뒤 사람이 LOST_SEC
이상 사라지면 STOP을 1회 발행한다(커튼이 움직이는 중 제스처 주체 소실).

순수 stdlib 구현(rknnlite/cv2 금지) — 보드 없이 단위테스트 가능.
"""

import os
from collections import deque
from math import hypot

# COCO keypoint indices
NOSE = 0
SH_L, SH_R = 5, 6
WR_L, WR_R = 9, 10

# --- 튜닝 파라미터 (어깨너비=1.0 단위, 시간=초) -----------------------------
KP_CONF = 0.3            # 어깨/손목 키포인트 최소 신뢰도
RAISE_MARGIN = 0.3       # 손목 인정 높이: y < 어깨y + 0.3*어깨너비 (든 손만)
TRAJ_SEC = 1.8           # 궤적 버퍼 길이 (HOLD_SEC 이상이어야 정지창이 참)
GAP_RESET_SEC = 0.35     # 샘플 공백이 이보다 길면 궤적 무효(리셋)
SWIPE_SEC = 0.6          # 스와이프 판정 창
SWIPE_DIST = 0.8         # 스와이프 수평 순변위 임계 (어깨너비의 0.8배)
SWIPE_AXIS_RATIO = 2.0   # |dx| > ratio*|dy| 일 때만 (수평 우세)
SWIPE_MIN_SPAN = 0.15    # 판정 창 안의 최소 시간 폭(2~3프레임 요동 방지)
SWIPE_MIN_SAMPLES = 3
# 정지(STOP)는 의도적 동작만 인정: 1.5초 유지(스와이프 0.6초보다 길어 먼저
# 발동되지 않음 → '들었다 쓸기' 충돌 방지) + 얼굴 근처면 제외(눈 비비기 차단)
HOLD_SEC = 1.5           # 정지(STOP) 유지 시간
HOLD_RADIUS = 0.15       # 정지 허용 이동 반경
FACE_GUARD = 0.55        # 손목이 코로부터 이보다 가까우면 정지 제외(얼굴 만지기)
REARM_RADIUS = 0.3       # 정지 재무장에 필요한 움직임 반경
REFRACTORY_SEC = 1.5     # 제스처 확정 후 불응기
LOST_SEC = 0.7           # 사람 미검출 → 안전 STOP까지의 시간
REACQUIRE_SEC = 0.5      # 소실 후 재검출 시 제스처 무시 시간(재진입 안정화)
IOU_STICKY = 0.2         # 동일인 유지에 필요한 최소 박스 IoU
TRAIL_SEC = 1.0          # 디버그 트레일 표시 길이

# 좌우 기준: 기본은 사용자(거울) 기준 — 사용자가 자기 오른쪽→왼쪽으로 쓸면
# 열림. 영상은 미러링되지 않으므로 이 동작은 영상 좌표 +x 방향이 된다.
# 화면 기준으로 쓰고 싶으면 CURTAIN_GESTURE_MIRROR=0.
MIRROR_ENV = "CURTAIN_GESTURE_MIRROR"
MIRROR_DEFAULT = True


def _iou(a, b):
    ix = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    iy = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class WristMotionClassifier:
    """이벤트형 손목 궤적 분류기. ``update(dets, now)`` → OPEN/CLOSE/STOP/None.

    dets: engine 디텍션 리스트 [{"box","score","keypoints"}], now: epoch 초.
    상태를 가지므로 프로파일 스왑 시 새로 생성해야 한다(profiles.make_classifier).
    """

    def __init__(self, mirror=None):
        if mirror is None:
            mirror = os.environ.get(
                MIRROR_ENV, "1" if MIRROR_DEFAULT else "0") != "0"
        self.mirror = mirror
        self._buf = deque()        # (t, nx, ny, px, py) — 정규화 + 픽셀 좌표
        self._box = None           # 추적 중인 person 박스 (동일인 고정용)
        self._wrist = None         # 추적 중인 손목 인덱스 (9 또는 10)
        self._last_seen = None     # 사람을 마지막으로 본 시각
        self._last_emit = None     # 마지막으로 발행한 라벨 (안전 STOP 판단용)
        self._refractory_until = 0.0
        self._ignore_until = 0.0
        self._stop_armed = True    # 정지(hold)는 움직임이 한 번 있어야 재무장
        self._near_face = False    # 추적 손목이 코 근처인가 (정지 제외용)
        self.status = "no person"  # 디버그 오버레이용 (cv2 렌더 가능한 ASCII)

    # --- 메인 엔트리 --------------------------------------------------------
    def update(self, dets, now):
        det = self._pick_person(dets)
        if det is None:
            self.status = "no person"
            self._wrist = None
            return self._check_lost(now)

        if self._last_seen is not None and now - self._last_seen > LOST_SEC:
            # 공백 후 재획득 — 잠시 판정을 멈춰 오인식 방지
            self._ignore_until = now + REACQUIRE_SEC
            self._clear(rearm=True)
        self._last_seen = now
        self._box = det["box"]

        sample = self._wrist_sample(det["keypoints"])
        if sample is None:
            self.status = "hand down"
            return None

        self._append(now, *sample)
        self._trim(now)

        if now < self._refractory_until or now < self._ignore_until:
            self.status = "cooldown %.1fs" % (
                max(self._refractory_until, self._ignore_until) - now)
            return None

        label = self._detect_swipe(now)
        if label is None:
            label = self._detect_hold(now)
        if label:
            self._emit(label, now)
        return label

    # --- 디버그/HUD ---------------------------------------------------------
    @property
    def trail(self):
        """최근 TRAIL_SEC 동안의 손목 픽셀 좌표 (오버레이용)."""
        if not self._buf:
            return []
        t_end = self._buf[-1][0]
        return [(int(px), int(py)) for (t, _nx, _ny, px, py) in self._buf
                if t_end - t <= TRAIL_SEC]

    # --- 내부 ----------------------------------------------------------------
    def _pick_person(self, dets):
        """직전 박스와 IoU 매칭으로 동일인 고정; 바뀌면 궤적 무효."""
        if not dets:
            return None
        if self._box is not None:
            best = max(dets, key=lambda d: _iou(self._box, d["box"]))
            if _iou(self._box, best["box"]) >= IOU_STICKY:
                return best
            self._clear(rearm=True)             # 다른 사람으로 전환
        return max(dets, key=lambda d: d["score"])

    def _wrist_sample(self, kp):
        """어깨 기준 정규화 손목 샘플 (nx, ny, px, py) — 들지 않았으면 None."""
        if kp[SH_L, 2] < KP_CONF or kp[SH_R, 2] < KP_CONF:
            return None                         # 스케일(어깨너비) 계산 불가
        sw = abs(float(kp[SH_L, 0]) - float(kp[SH_R, 0])) + 1e-6
        sh_x = (float(kp[SH_L, 0]) + float(kp[SH_R, 0])) / 2.0
        sh_y = (float(kp[SH_L, 1]) + float(kp[SH_R, 1])) / 2.0
        y_max = sh_y + RAISE_MARGIN * sw        # 이 위로 든 손목만 인정

        def ok(i):
            return kp[i, 2] >= KP_CONF and kp[i, 1] < y_max

        # 같은 손목을 계속 추적 — 좌/우가 프레임마다 바뀌면 궤적이 튄다
        if self._wrist is not None and ok(self._wrist):
            idx = self._wrist
        else:
            cand = [i for i in (WR_L, WR_R) if ok(i)]
            if not cand:
                self._wrist = None
                return None
            idx = max(cand, key=lambda i: kp[i, 2])
            if self._wrist is not None and idx != self._wrist:
                self._clear(rearm=True)         # 반대 손으로 전환 → 궤적 무효
            self._wrist = idx

        px, py = float(kp[idx, 0]), float(kp[idx, 1])
        # 손목이 코(얼굴) 근처인지 — 눈 비비기/얼굴 만지기를 정지로 오인 방지
        self._near_face = (kp[NOSE, 2] >= KP_CONF and
                           hypot(px - float(kp[NOSE, 0]),
                                 py - float(kp[NOSE, 1])) < FACE_GUARD * sw)
        # 몸통(어깨 중심) 상대 좌표 — 걷기 등 몸 전체 이동을 상쇄
        return (px - sh_x) / sw, (py - sh_y) / sw, px, py

    def _append(self, now, nx, ny, px, py):
        if self._buf and now - self._buf[-1][0] > GAP_RESET_SEC:
            self._clear(rearm=True)             # 공백을 넘긴 궤적은 무효
        self._buf.append((now, nx, ny, px, py))

    def _trim(self, now):
        while self._buf and now - self._buf[0][0] > TRAJ_SEC:
            self._buf.popleft()

    def _detect_swipe(self, now):
        win = [s for s in self._buf if now - s[0] <= SWIPE_SEC]
        if (len(win) < SWIPE_MIN_SAMPLES or
                win[-1][0] - win[0][0] < SWIPE_MIN_SPAN):
            self.status = "tracking"
            return None
        dx = win[-1][1] - win[0][1]
        dy = win[-1][2] - win[0][2]
        self.status = "tracking dx=%+.2f" % dx
        if abs(dx) < SWIPE_DIST or abs(dx) <= SWIPE_AXIS_RATIO * abs(dy):
            return None
        # 영상 좌표 +x = 사용자 기준 우→좌 (카메라 영상은 미러링되지 않음)
        if self.mirror:
            return "OPEN" if dx > 0 else "CLOSE"
        return "OPEN" if dx < 0 else "CLOSE"

    def _detect_hold(self, now):
        if self._near_face:
            return None                         # 얼굴 만지기는 정지로 보지 않음
        win = [s for s in self._buf if now - s[0] <= HOLD_SEC]
        if len(win) < 4 or win[-1][0] - win[0][0] < HOLD_SEC * 0.9:
            return None
        cx = sum(s[1] for s in win) / len(win)
        cy = sum(s[2] for s in win) / len(win)
        radius = max(hypot(s[1] - cx, s[2] - cy) for s in win)
        if radius <= HOLD_RADIUS:
            return "STOP" if self._stop_armed else None
        if radius >= REARM_RADIUS:
            self._stop_armed = True             # 충분히 움직였음 → 정지 재무장
        return None

    def _check_lost(self, now):
        """사람 소실 처리: 커튼을 움직여 놓고 사라졌으면 STOP 1회."""
        self._trim(now)
        if self._last_seen is not None and now - self._last_seen > LOST_SEC:
            moving = self._last_emit in ("OPEN", "CLOSE")
            self._last_seen = None
            self._box = None
            self._clear(rearm=True)
            if moving:
                self._emit("STOP", now)
                self.status = "lost -> STOP"
                return "STOP"
        return None

    def _emit(self, label, now):
        self._last_emit = label
        self._refractory_until = now + REFRACTORY_SEC
        if label == "STOP":
            self._stop_armed = False            # 계속 들고 있어도 반복 발행 금지
        self._clear(rearm=(label != "STOP"))

    def _clear(self, rearm):
        self._buf.clear()
        if rearm:
            self._stop_armed = True
