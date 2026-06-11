"""gesture_motion(손목 움직임 제스처) 단위 테스트 — 보드/카메라 불필요.

합성 궤적으로 판정·디바운싱·안전 로직을 검증한다:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gesture_motion as gm                                  # noqa: E402
from gesture_motion import WristMotionClassifier             # noqa: E402

SW = 100.0          # 어깨너비(픽셀)
CX = 320.0          # 어깨 중심 x
SH_Y = 200.0        # 어깨 y
RAISED_Y = SH_Y - 20.0   # 어깨 위로 든 손목 높이
DT = 0.05           # 프레임 간격(20FPS 가정)


HOLD = gm.HOLD_SEC + 0.4   # 정지 판정에 충분한 유지 시간(상수 변경에 자동 추종)


def det(wx, wy, wrist="r", score=0.9, box=None, wrist_conf=0.8, nose=None):
    """전신 17키포인트 합성 디텍션 (어깨 2점 + 손목 1점 + 선택적 코)."""
    kp = np.zeros((17, 3), np.float32)
    kp[gm.SH_L] = (CX - SW / 2, SH_Y, 0.9)
    kp[gm.SH_R] = (CX + SW / 2, SH_Y, 0.9)
    kp[gm.WR_R if wrist == "r" else gm.WR_L] = (wx, wy, wrist_conf)
    if nose is not None:
        kp[gm.NOSE] = (nose[0], nose[1], 0.9)
    if box is None:
        box = (CX - 100, SH_Y - 80, CX + 100, SH_Y + 250)
    return {"box": np.array(box, np.float32), "score": score, "keypoints": kp}


def feed(c, xs, t0=0.0, dt=DT, wy=RAISED_Y, **kw):
    """x 좌표 시퀀스를 프레임으로 공급, (이벤트 목록, 종료 시각) 반환."""
    events, t = [], t0
    for x in xs:
        e = c.update([det(x, wy, **kw)], t)
        if e:
            events.append(e)
        t += dt
    return events, t


def hold(c, x, secs, t0, wy=RAISED_Y, **kw):
    n = int(round(secs / DT))
    return feed(c, [x] * n, t0=t0, wy=wy, **kw)


class SwipeTest(unittest.TestCase):
    def test_user_right_to_left_is_open(self):
        # 사용자 기준 우→좌 = 영상 좌표 +x (카메라 영상은 미러링 안 됨)
        c = WristMotionClassifier(mirror=True)
        events, _ = feed(c, np.linspace(200, 440, 12))   # 2.4*SW in 0.55s
        self.assertEqual(events, ["OPEN"])

    def test_user_left_to_right_is_close(self):
        c = WristMotionClassifier(mirror=True)
        events, _ = feed(c, np.linspace(440, 200, 12))
        self.assertEqual(events, ["CLOSE"])

    def test_mirror_off_flips_direction(self):
        c = WristMotionClassifier(mirror=False)
        events, _ = feed(c, np.linspace(200, 440, 12))
        self.assertEqual(events, ["CLOSE"])

    def test_mirror_env_default(self):
        with mock.patch.dict(os.environ, {gm.MIRROR_ENV: "0"}):
            self.assertFalse(WristMotionClassifier().mirror)
        with mock.patch.dict(os.environ, {gm.MIRROR_ENV: "1"}):
            self.assertTrue(WristMotionClassifier().mirror)

    def test_short_swipe_does_not_fire(self):
        c = WristMotionClassifier(mirror=True)
        events, _ = feed(c, np.linspace(300, 380, 12))   # 0.8*SW < SWIPE_DIST
        self.assertEqual(events, [])

    def test_diagonal_motion_does_not_fire(self):
        # 수직 성분이 큰 움직임(팔 들어올리기 등)은 스와이프가 아님
        c = WristMotionClassifier(mirror=True)
        t, events = 0.0, []
        for x, y in zip(np.linspace(200, 440, 12),
                        np.linspace(SH_Y + 25, SH_Y - 175, 12)):
            e = c.update([det(x, y)], t)
            if e:
                events.append(e)
            t += DT
        self.assertEqual(events, [])

    def test_refractory_blocks_immediate_second_swipe(self):
        c = WristMotionClassifier(mirror=True)
        events, t = feed(c, np.linspace(200, 440, 12))           # OPEN
        ev2, _ = feed(c, np.linspace(440, 200, 12), t0=t)        # 곧바로 반대
        self.assertEqual(events, ["OPEN"])
        self.assertEqual(ev2, [])                                # 불응기로 무시

    def test_gap_resets_trajectory(self):
        # 0.5초 공백 사이의 큰 변위는 스와이프로 합산되면 안 됨
        c = WristMotionClassifier(mirror=True)
        c.update([det(200, RAISED_Y)], 0.0)
        events = []
        t = 0.5                                                  # > GAP_RESET_SEC
        for x in [440] * 6:
            e = c.update([det(x, RAISED_Y)], t)
            if e:
                events.append(e)
            t += DT
        self.assertEqual(events, [])


class HoldStopTest(unittest.TestCase):
    def test_raise_and_hold_is_stop(self):
        c = WristMotionClassifier(mirror=True)
        events, _ = hold(c, CX + 30, HOLD, t0=0.0)
        self.assertEqual(events, ["STOP"])

    def test_brief_pause_before_swipe_is_not_stop(self):
        # 손 올리고 0.6초 멈칫 후 스와이프 → STOP이 먼저 나면 안 됨 (열기/닫기 우선)
        c = WristMotionClassifier(mirror=True)
        ev1, t = hold(c, 200.0, 0.6, t0=0.0)                     # 들고 잠깐 멈칫
        ev2, _ = feed(c, np.linspace(200, 440, 12), t0=t)        # 이어서 우→좌 스와이프
        self.assertEqual(ev1, [])                                # 멈칫은 STOP 아님
        self.assertEqual(ev2, ["OPEN"])

    def test_continued_hold_does_not_repeat_stop(self):
        c = WristMotionClassifier(mirror=True)
        ev1, t = hold(c, CX + 30, HOLD, t0=0.0)                  # STOP
        ev2, _ = hold(c, CX + 30, 3.0, t0=t)                     # 계속 유지
        self.assertEqual(ev1, ["STOP"])
        self.assertEqual(ev2, [])                # 재무장 전엔 반복 발행 금지

    def test_hand_at_face_is_not_stop(self):
        # 눈 비비기 등 손목이 코(얼굴) 근처면 들고 멈춰도 STOP 제외
        c = WristMotionClassifier(mirror=True)
        events, _ = hold(c, CX + 5, HOLD + 0.5, t0=0.0,
                         wy=RAISED_Y, nose=(CX, RAISED_Y))
        self.assertEqual(events, [])

    def test_lower_and_raise_rearms_stop(self):
        c = WristMotionClassifier(mirror=True)
        ev1, t = hold(c, CX + 30, HOLD, t0=0.0)                  # STOP
        ev2, t = hold(c, CX + 30, 1.0, t0=t, wy=SH_Y + 200.0)    # 손 내림(미인정)
        ev3, t = hold(c, CX + 30, HOLD, t0=t)                    # 다시 들고 유지
        self.assertEqual(ev1, ["STOP"])
        self.assertEqual(ev2, [])
        self.assertEqual(ev3, ["STOP"])

    def test_lowered_hand_never_fires(self):
        # 어깨 아래(내린 손)의 정지/이동은 무시 — 걷기 오인식 방지
        c = WristMotionClassifier(mirror=True)
        ev1, t = hold(c, CX + 30, 1.5, t0=0.0, wy=SH_Y + 200.0)
        ev2, _ = feed(c, np.linspace(200, 440, 12), t0=t, wy=SH_Y + 200.0)
        self.assertEqual(ev1 + ev2, [])

    def test_low_confidence_wrist_ignored(self):
        c = WristMotionClassifier(mirror=True)
        events, _ = feed(c, np.linspace(200, 440, 12), wrist_conf=0.1)
        self.assertEqual(events, [])


class SafetyTest(unittest.TestCase):
    def test_person_lost_while_moving_fires_stop_once(self):
        c = WristMotionClassifier(mirror=True)
        ev1, t = feed(c, np.linspace(200, 440, 12))              # OPEN(이동 시작)
        events = []
        for _ in range(30):                                      # 1.5초간 미검출
            e = c.update([], t)
            if e:
                events.append(e)
            t += DT
        self.assertEqual(ev1, ["OPEN"])
        self.assertEqual(events, ["STOP"])                       # 정확히 1회

    def test_person_lost_after_stop_is_silent(self):
        c = WristMotionClassifier(mirror=True)
        ev1, t = hold(c, CX + 30, HOLD, t0=0.0)                  # STOP(정지 상태)
        events = []
        for _ in range(30):
            e = c.update([], t)
            if e:
                events.append(e)
            t += DT
        self.assertEqual(ev1, ["STOP"])
        self.assertEqual(events, [])             # 움직이는 중이 아니면 침묵

    def test_no_command_yet_lost_is_silent(self):
        c = WristMotionClassifier(mirror=True)
        _, t = hold(c, CX + 30, 0.3, t0=0.0)                     # 아무 확정 없음
        events = []
        for _ in range(30):
            e = c.update([], t)
            if e:
                events.append(e)
            t += DT
        self.assertEqual(events, [])

    def test_reacquire_grace_blocks_instant_gesture(self):
        # 소실 후 재검출 직후의 궤적은 REACQUIRE_SEC 동안 무시
        c = WristMotionClassifier(mirror=True)
        _, t = hold(c, CX + 30, 0.3, t0=0.0)
        t += gm.LOST_SEC + 0.2                                   # 사람 소실 공백
        events, _ = feed(c, np.linspace(200, 440, 8), t0=t)      # 곧바로 스와이프
        self.assertEqual(events, [])


class TrackingTest(unittest.TestCase):
    def test_person_switch_resets_trajectory(self):
        # 추적 대상이 다른 사람으로 바뀌면 변위가 합산되면 안 됨
        c = WristMotionClassifier(mirror=True)
        box_a = (150, 100, 350, 450)
        box_b = (600, 100, 800, 450)
        t = 0.0
        events = []
        for x in np.linspace(200, 260, 5):                       # A: 0.6*SW 이동
            e = c.update([det(x, RAISED_Y, box=box_a)], t)
            if e:
                events.append(e)
            t += DT
        for x in np.linspace(460, 520, 5):                       # B: 0.6*SW 이동
            e = c.update([det(x, RAISED_Y, box=box_b, score=0.95)], t)
            if e:
                events.append(e)
            t += DT
        self.assertEqual(events, [])             # 합산 시 3.2*SW → 오발사

    def test_wrist_switch_resets_trajectory(self):
        # 오른손목 → 왼손목 전환 시 두 손의 변위가 이어지면 안 됨
        c = WristMotionClassifier(mirror=True)
        t = 0.0
        events = []
        for x in np.linspace(200, 260, 5):
            e = c.update([det(x, RAISED_Y, wrist="r")], t)
            if e:
                events.append(e)
            t += DT
        for x in np.linspace(460, 520, 5):
            e = c.update([det(x, RAISED_Y, wrist="l")], t)
            if e:
                events.append(e)
            t += DT
        self.assertEqual(events, [])

    def test_trail_exposed_for_overlay(self):
        c = WristMotionClassifier(mirror=True)
        feed(c, np.linspace(300, 340, 5))
        trail = c.trail
        self.assertEqual(len(trail), 5)
        self.assertEqual(trail[-1], (340, int(RAISED_Y)))


if __name__ == "__main__":
    unittest.main()
