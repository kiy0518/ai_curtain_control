# 08. 배포 · 운영 · 버전 관리

> Phase 7 · 목표버전 v1.0.0 · 상태: 일부(git/태그 운용 중)

## 배포 / 자동 실행  ★필수 · 조기 적용(v0.2.0)
> **요구사항: 전원 ON 시 자동 실행** — 보드에 전원만 넣으면 카메라·추론·웹서버가 자동 기동.
> 다른 Phase에 의존하지 않으므로 가능한 한 빨리 적용.

- [ ] systemd 서비스(`deploy/ai-curtain.service`): 부팅 시 자동 실행, 비정상 종료 시 자동 재시작
- [ ] 카메라/NPU/시리얼 준비 대기(의존성·재시도, AE 수렴 고려)
- [ ] 설치 스크립트(`deploy/install.sh`): 의존성·서비스 등록 일괄
- [ ] 환경 설정 분리(`config.env`: 포트/모델/imgsz/시리얼/임계값)
- [ ] 부팅~서비스 기동 시간 확인, 실패 시 로그로 원인 추적

### systemd 유닛 예시 (`deploy/ai-curtain.service`)
```ini
[Unit]
Description=AI Curtain Control
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=radxa
WorkingDirectory=/home/radxa/Documents/ai_curtain_control
# 카메라 ISP/NPU 준비 대기(여유)
ExecStartPre=/bin/sleep 8
ExecStart=/usr/bin/python3 serve.py --model models/hand_pose_640.rknn --imgsz 640 --conf 0.3
Restart=always
RestartSec=3
# 환경설정(config.env) 사용 시
EnvironmentFile=-/home/radxa/Documents/ai_curtain_control/config.env

[Install]
WantedBy=multi-user.target
```
```bash
sudo cp deploy/ai-curtain.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-curtain     # 부팅 자동실행 + 즉시 시작
systemctl status ai-curtain                # 상태 확인
journalctl -u ai-curtain -f                # 로그
```
> ※ 인자(모델/imgsz)는 추후 `config.env`로 빼서 ExecStart 단순화 권장.

## 운영 / 모니터링
- [ ] 구조적 로깅(파일 로테이션) + 레벨
- [ ] 헬스체크 엔드포인트(`/healthz`) + 시스템 지표(CPU/메모리/온도/FPS)
- [ ] 장애 알림(선택: 메일/푸시)

## OTA / 업데이트
- [ ] `git pull` 기반 업데이트 + 서비스 재시작 스크립트
- [ ] 모델 핫스왑(대시보드 관리자에서 .rknn 교체)

## 버전 관리 (운용 규칙)
- SemVer `v0.x` (1.0 전 개발). Phase 완료 → MINOR +1, 버그픽스 → PATCH.
- 각 버전: **annotated 태그 + GitHub 릴리스**, 노트에 `생성시각 / 진행내용 / 미해결 승계` 포함.
- `00_progress.md` 의 Phase·버전표를 릴리스마다 갱신.
- 기능 개발: `feat/<phase>-<name>` 브랜치 → `main` 머지.
- 예:
  ```bash
  git switch -c feat/phase1-motor
  # ...작업/커밋...
  git switch main && git merge --no-ff feat/phase1-motor
  git tag -a v0.2.0 -m "v0.2.0 ... 시각/진행/승계 ..." && git push origin main v0.2.0
  ```

## 현재까지 완료
- [x] git 저장소 + `main` + 원격(GitHub)
- [x] v0.1.0 태그 + 릴리스(노트: 시각/진행/승계)
- [ ] 자동실행/설치 스크립트 (미착수)
