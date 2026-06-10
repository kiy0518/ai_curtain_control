# 08. 배포 · 운영 · 버전 관리

> Phase 6 · 목표버전 v1.0.0 · 상태: 일부(git/태그 운용 중)
> ※ 전원 ON 자동실행은 **Phase M(모터 준비 후)** 에 실현 — 아래 systemd 설정을 그때 적용.

## 배포 / 자동 실행  ★요구사항 · 실현 시점: **Phase M(모터 준비 후)**
> **요구사항: 전원 ON 시 자동 실행** — 보드에 전원만 넣으면 카메라·추론·웹서버(+모터제어)가 자동 기동.
> 모터 컨트롤러 연동이 끝나는 시점에 함께 적용(사용자 요청). 아래 유닛은 준비된 설정.

- [ ] systemd 서비스(`deploy/ai-curtain.service`): 부팅 시 자동 실행, 비정상 종료 시 자동 재시작
- [ ] 카메라/NPU/시리얼 준비 대기(의존성·재시도, AE 수렴 고려)
- [ ] 설치 스크립트(`deploy/install.sh`): 의존성·서비스 등록 일괄
- [ ] 환경 설정 분리(`config.env`: 포트/모델/imgsz/시리얼/임계값)
- [ ] 부팅~서비스 기동 시간 확인, 실패 시 로그로 원인 추적

### systemd 유닛 예시 (`deploy/ai-curtain.service`)
```ini
[Unit]
Description=AI Curtain Control
# 로컬(카메라/NPU/제스처)은 네트워크 불필요 → network-online 의존 제거(부팅 지연 방지).
# 원격(cloudflared 등)은 별도 유닛에서 network-online 의존.

[Service]
Type=simple
User=radxa
SupplementaryGroups=dialout        # 시리얼(/dev/ttyS*) 접근 (radxa는 기본 미포함)
WorkingDirectory=/home/radxa/Documents/ai_curtain_control
ExecStart=/usr/bin/python3 serve.py --model models/hand_pose_640.rknn --imgsz 640 --conf 0.3
Restart=always
RestartSec=3
StartLimitIntervalSec=60
StartLimitBurst=5                  # 의존성 영구 실패 시 무한 크래시루프 방지
EnvironmentFile=-/home/radxa/Documents/ai_curtain_control/config.env

[Install]
WantedBy=multi-user.target
```
> 카메라 ISP/NPU/시리얼 준비는 고정 `sleep`이 아니라 **앱이 재시도(없으면 대기)** 로 처리(브리틀한 타이밍 의존 회피). AE 수렴은 스트림이 자연 처리.
> 시리얼 포트가 udev `.device` 유닛으로 잡히면 `After=`/`BindsTo=dev-ttyS3.device` 추가 검토.
```bash
sudo cp deploy/ai-curtain.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-curtain     # 부팅 자동실행 + 즉시 시작
systemctl status ai-curtain                # 상태 확인
journalctl -u ai-curtain -f                # 로그
```
> ※ 인자(모델/imgsz)는 추후 `config.env`로 빼서 ExecStart 단순화 권장.

## 운영 / 모니터링
- [x] 구조적 로깅(`logs/curtain.log`, RotatingFileHandler) + 콘솔
- [x] 헬스체크 `GET /healthz`(무인증) — status/uptime/camera/fps/profile
- [x] 시스템 지표 대시보드 표시(load/메모리/온도/FPS) — `/api/state`
- [ ] 장애 알림(선택: 메일/푸시)

## 설정 / 배포
- [x] `config.env`(CURTAIN_*) 환경변수 분리 + app.py env 기본값
- [x] `deploy/install.sh`(opencv·cloudflared·config.env), `deploy/ai-curtain.service`
- [x] `deploy/update.sh`(git pull + systemctl restart) — OTA 형태
- [x] 모델 핫스왑(대시보드 관리자에서 프로파일 런타임 전환)
- [ ] 자동실행 enable 은 Phase M(모터 준비) 시 (유닛 파일은 준비 완료)

## 버전 관리 (운용 규칙)
- SemVer `v0.x` (1.0 전 개발). Phase 완료 → MINOR +1, 버그픽스 → PATCH.
- 각 버전: **annotated 태그 + GitHub 릴리스**, 노트에 `생성시각 / 진행내용 / 미해결 승계` 포함.
- `00_progress.md` 의 Phase·버전표를 릴리스마다 갱신.
- 기능 개발: `feat/<phase>-<name>` 브랜치 → `main` 머지.
- 예:
  ```bash
  git switch -c feat/phase1-gesture     # 예: Phase 1(제스처) → v0.2.0
  # ...작업/커밋...
  git switch main && git merge --no-ff feat/phase1-gesture
  git tag -a v0.2.0 -m "v0.2.0 ... 시각/진행/승계 ..." && git push origin main v0.2.0
  ```

## 현재까지 완료
- [x] git 저장소 + `main` + 원격(GitHub)
- [x] v0.1.0 태그 + 릴리스(노트: 시각/진행/승계)
- [ ] 자동실행/설치 스크립트 (미착수)
