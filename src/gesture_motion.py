"""Wrist-motion gestures from COCO 17-keypoint body pose (far-range control).

손모양(hand_near)은 ~2m 이내에서만 동작하므로, 원거리에서는 전신 모델의
**손목 키포인트**로 판정한다. 좌표를 어깨너비로 정규화하므로 거리와 무관하게
같은 임계값이 적용된다(거리 불변).

  * OPEN  (열림): 손을 어깨 높이 이상으로 들고 우→좌 쓸기 (사용자 기준)
  * CLOSE (닫힘): 손을 어깨 높이 이상으로 들고 좌→우 쓸기
  * STOP  (정지): 양팔 X 교차 — 두 손목을 어깨 사이 가슴 높이로 모은 자세를
                  HOLD_SEC 동안 유지 (body_far의 X 정지와 동일한 자세)

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
SH_L, SH_R = 5, 6
WR_L, WR_R = 9, 10

# --- 튜닝 파라미터 (어깨너비=1.0 단위, 시간=초) -----------------------------
KP_CONF = 0.3            # 어깨/손목 키포인트 최소 신뢰도
RAISE_MARGIN = 0.3       # 손목 인정 높이: y < 어깨y + 0.3*어깨너비 (든 손만)
TRAJ_SEC = 1.2           # 궤적 버퍼 길이
GAP_RESET_SEC = 0.35     # 샘플 공백이 이보다 길면 궤적 무효(리셋)
SWIPE_SEC = 0.6          # 스와이프 판정 창
SWIPE_DIST = 0.8         # 스와이프 수평 순변위 임계 (어깨너비의 0.8배)
SWIPE_AXIS_RATIO = 2.0   # |dx| > ratio*|dy| 일 때만 (수평 우세)
SWIPE_MIN_SPAN = 0.15    # 판정 창 안의 최소 시간 폭(2~3프레임 요동 방지)
SWIPE_MIN_SAMPLES = 3
# 정지(STOP)=양팔 X 교차 자세를 이 시간 유지하면 확정 (대시보드에서 조정 가능)
HOLD_SEC = 1.5           # X 자세 유지 시간
CROSS_CHEST = 0.6        # X: |손목y - 어깨y| < 0.6*어깨너비 (가슴 높이)
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
    """이벤트형 손목 분류기. ``update(dets, now)`` → OPEN/CLOSE/STOP/None.

    dets: engine 디텍션 리스트 [{"box","score","keypoints"}], now: epoch 초.
    상태를 가지므로 프로파일 스왑 시 새로 생성해야 한다(profiles.make_classifier).
    """

    def __init__(self, mirror=None, hold_sec=None, refractory_sec=None):
        if mirror is None:
            mirror = os.environ.get(
                MIRROR_ENV, "1" if MIRROR_DEFAULT else "0") != "0"
        self.mirror = mirror
        # 런타임 조정: X 정지 유지 시간 / 불응 시간 / 스와이프 길이 (대시보드 설정)
        self.hold_sec = HOLD_SEC
        self.refractory_sec = REFRACTORY_SEC
        self.swipe_dist = SWIPE_DIST
        self.set_timing(hold_sec, refractory_sec)
        self._buf = deque()        # (t, nx, ny, px, py) — 정규화 + 픽셀 좌표
        self._box = None           # 추적 중인 person 박스 (동일인 고정용)
        self._wrist = None         # 추적 중인 손목 인덱스 (9 또는 10)
        self._cross_since = None    # X 자세를 처음 연속 인식한 시각
        self._last_seen = None     # 사람을 마지막으로 본 시각
        self._last_emit = None     # 마지막으로 발행한 라벨 (안전 STOP 판단용)
        self._refractory_until = 0.0
        self._ignore_until = 0.0
        self.status = "no person"  # 디버그 오버레이용 (cv2 렌더 가능한 ASCII)

    def set_timing(self, hold_sec=None, refractory_sec=None, swipe_dist=None):
        """X 정지 유지 시간 / 불응 시간 / 스와이프 길이를 런타임에 조정."""
        if hold_sec:
            self.hold_sec = float(hold_sec)
        if refractory_sec:
            self.refractory_sec = float(refractory_sec)
        if swipe_dist:
            self.swipe_dist = float(swipe_dist)

    # --- 메인 엔트리 --------------------------------------------------------
    def update(self, dets, now):
        det = self._pick_person(dets)
        if det is None:
            self.status = "no person"
            self._wrist = None
            self._cross_since = None
            return self._check_lost(now)

        if self._last_seen is not None and now - self._last_seen > LOST_SEC:
            # 공백 후 재획득 — 잠시 판정을 멈춰 오인식 방지
            self._ignore_until = now + REACQUIRE_SEC
            self._clear()
        self._last_seen = now
        self._box = det["box"]
        kp = det["keypoints"]

        cooling = now < self._refractory_until or now < self._ignore_until

        # 정지(양팔 X 교차) — 자세 기반(궤적 무관). 쿨다운 중엔 타이머만 리셋.
        if cooling:
            self._cross_since = None
        elif self._detect_cross(kp, now):
            self._emit("STOP", now)
            self.status = "X stop"
            return "STOP"

        # 열림/닫힘(쓸기) — 든 손목 궤적
        sample = self._wrist_sample(kp)
        if sample is None:
            self.status = "hand down"
            return None
        self._append(now, *sample)
        self._trim(now)
        if cooling:
            self.status = "cooldown %.1fs" % (
                max(self._refractory_until, self._ignore_until) - now)
            return None
        label = self._detect_swipe(now)
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
            self._clear()                       # 다른 사람으로 전환
        return max(dets, key=lambda d: d["score"])

    def _detect_cross(self, kp, now):
        """양팔 X 교차 자세를 HOLD_SEC 동안 유지하면 True (body_far와 동일 기준):
        두 손목이 어깨 사이(가로) + 가슴 높이(세로)에 모인 자세."""
        if (kp[SH_L, 2] < KP_CONF or kp[SH_R, 2] < KP_CONF or
                kp[WR_L, 2] < KP_CONF or kp[WR_R, 2] < KP_CONF):
            self._cross_since = None
            return False
        sw = abs(float(kp[SH_L, 0]) - float(kp[SH_R, 0])) + 1e-6
        sh_y = (float(kp[SH_L, 1]) + float(kp[SH_R, 1])) / 2.0
        x_lo = min(float(kp[SH_L, 0]), float(kp[SH_R, 0]))
        x_hi = max(float(kp[SH_L, 0]), float(kp[SH_R, 0]))
        chest = (abs(float(kp[WR_L, 1]) - sh_y) < CROSS_CHEST * sw and
                 abs(float(kp[WR_R, 1]) - sh_y) < CROSS_CHEST * sw)
        between = (x_lo < float(kp[WR_L, 0]) < x_hi and
                   x_lo < float(kp[WR_R, 0]) < x_hi)
        if not (chest and between):
            self._cross_since = None
            return False
        if self._cross_since is None:
            self._cross_since = now
        return now - self._cross_since >= self.hold_sec

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
                self._clear()                   # 반대 손으로 전환 → 궤적 무효
            self._wrist = idx

        px, py = float(kp[idx, 0]), float(kp[idx, 1])
        # 몸통(어깨 중심) 상대 좌표 — 걷기 등 몸 전체 이동을 상쇄
        return (px - sh_x) / sw, (py - sh_y) / sw, px, py

    def _append(self, now, nx, ny, px, py):
        if self._buf and now - self._buf[-1][0] > GAP_RESET_SEC:
            self._clear()                       # 공백을 넘긴 궤적은 무효
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
        if abs(dx) < self.swipe_dist or abs(dx) <= SWIPE_AXIS_RATIO * abs(dy):
            return None
        # 영상 좌표 +x = 사용자 기준 우→좌 (카메라 영상은 미러링되지 않음)
        if self.mirror:
            return "OPEN" if dx > 0 else "CLOSE"
        return "OPEN" if dx < 0 else "CLOSE"

    def _check_lost(self, now):
        """사람 소실 처리: 커튼을 움직여 놓고 사라졌으면 STOP 1회."""
        self._trim(now)
        if self._last_seen is not None and now - self._last_seen > LOST_SEC:
            moving = self._last_emit in ("OPEN", "CLOSE")
            self._last_seen = None
            self._box = None
            self._clear()
            if moving:
                self._emit("STOP", now)
                self.status = "lost -> STOP"
                return "STOP"
        return None

    def _emit(self, label, now):
        self._last_emit = label
        self._refractory_until = now + self.refractory_sec
        self._clear()

    def _clear(self):
        self._buf.clear()
        self._cross_since = None
