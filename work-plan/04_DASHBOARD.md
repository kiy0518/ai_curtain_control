# 04. 대시보드 (웹 / 모바일)

> Phase 2 · 목표버전 v0.3.0 · 상태: 미착수

## 목표
실시간 영상 + 커튼 상태 + 설정/스케줄을 한 화면에서. 웹 우선, **PWA로 모바일 대응**(코드 1벌).

> ⚠️ **모터 미연동 단계**: 커튼 제어 버튼(열기/닫기/정지)은 우선 **placeholder(UI만, 동작 비활성)** 로
> 만들고, Phase M(모터 시리얼) 완료 시 실제 명령에 연결한다. 상태 표시도 mock → 실제로 전환.

## 화면/기능
### 사용자 화면
- [ ] 실시간 영상 스트리밍(현 MJPEG → 추후 WebRTC 검토)
- [ ] 현재 커튼 상태 + 수동 버튼(열기/닫기/정지) — *모터 연동 전 placeholder*
- [ ] 개폐 스케줄 관리(추가/수정/삭제) — `05_*`
- [ ] 제스처 인식 ON/OFF 토글
- [ ] **오프라인/오류 상태 표시**: 카메라 오프라인·NPU 실패·모터 미연결·WebSocket 끊김 시 명확히 표기 + 자동 재연결

### 관리자 화면
- [ ] 입력 이미지 크기 선택(224/320/640)
- [ ] 모델 선택(.rknn 목록) + 적용(서버 재시작/리로드)
- [ ] 신뢰도(conf)·제스처 임계값 조정
- [ ] **위치(좌표)·시간대 설정**(일출/일몰 스케줄용 — `05_*`)
- [ ] 시스템 상태(NPU/CPU/메모리/온도, FPS, 로그)

> 부분 개폐(%)는 위치 피드백 확정 전까지 **표시/의도값(시간 추정)** 으로만 — 정밀 위치는 Phase M 이후.

## 설계
- 서버: **FastAPI + uvicorn**, WebSocket으로 상태/이벤트 푸시(프레임은 보내지 않음).
- 라우트: `/`(대시보드), `/api/state`, `/api/control`, `/api/settings`, `/api/schedules`, `/stream.mjpg`, `/ws`.
- **스트리밍 부하 주의**: JPEG는 **단일 공유 인코더**(현 `streaming.py`의 latest-slot 방식 유지)로 1회 인코딩 후 모든 클라이언트에 fan-out. 클라이언트별 재인코딩 금지, 동시 시청자 수 제한, 느린 소비자는 프레임 드롭.
- ⚠️ **서빙 경로에서 `ultralytics`/torch import 금지**(torch가 이 보드에서 SIGILL). 추론은 `rknnlite` 직접 호출만.
- 프론트: 경량 HTML/JS + Service Worker(PWA).

## 데이터 모델 (SQLite) — 공통 스키마
> `04/05/07` 이 각자 만들지 않도록 여기서 1벌로 정의(초안).
- `settings(key, value)` — imgsz/모델/conf/위치/TZ 등
- `schedules(id, name, action, time/cron, sun_offset, days, enabled)`
- `profiles(id, name, prefs_json)` — 개인 선호(개폐량 등)
- `users(id, username, pw_hash, role)` — 계정(단일/다중 정책에 따라, `07_*`)
- `events(ts, type, detail)` — 동작/오류 이력(감사 로그)
- [ ] 마이그레이션 전략(간단 버전 테이블 또는 Alembic)

## 작업
- [ ] FastAPI 스캐폴딩 + 기존 스트리밍 통합(공유 인코더)
- [ ] 상태/제어 REST + WebSocket + 오프라인/오류 상태
- [ ] SQLite 스키마/마이그레이션
- [ ] 대시보드 UI(사용자/관리자 탭)
- [ ] PWA(manifest+SW), 모바일 레이아웃

## 의존성
- 제어(`02_*`)·스케줄(`05_*`) API 선행 또는 mock으로 병행.
- 인증(`07_*`)은 관리자 화면 노출 전 적용 권장.
