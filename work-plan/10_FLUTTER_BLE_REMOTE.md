# 10. 유저 리모컨 앱 (Flutter + Bluetooth BLE)

> Phase 7 · 목표버전 v1.1.0 · 상태: **진행중** (보드 BLE 서버 PoC 동작 / Flutter 스캐폴드 완료)

### 확정된 결정 (사용자)
- 앱 코드 위치: **이 저장소 `app_flutter/`**
- BLE 서버 라이브러리: **`bless`** (설치·광고 검증됨)
- 인증: **BLE 본딩(페어링)만**
- 플랫폼: **Android APK 우선** (iOS 추후)

### 진행 (이번)
- [x] 보드 블루투스 확인(hci0 USB AIC, BlueZ 1.22) + `bless` 설치
- [x] `src/ble_server.py` — Curtain 서비스(Command write / Status notify) + `CurtainController` 연동, 광고 동작 확인("AI-Curtain")
- [x] `app.py --ble`(또는 `CURTAIN_BLE=1`) 로 대시보드와 같은 프로세스에서 기동
- [x] Flutter 스캐폴드 `app_flutter/`(pubspec, lib/main.dart, 권한, README)
- [ ] 실제 폰 연결 테스트(nRF Connect/앱) + 본딩
- [ ] APK 빌드(PC) · 자동재연결 · 다중 클라이언트 정책

손짓/대시보드 외에, **휴대폰 Flutter 앱**으로 커튼을 직접 제어하는 리모컨. 근접 제어는
**BLE(Bluetooth Low Energy)** 로 연결한다. (웹 대시보드/cloudflared 는 원격·관리용,
BLE 리모컨은 집 안 근거리 즉시 제어용 — 역할 분리.)

```
[Flutter 앱 (BLE Central)]  ←BLE GATT→  [ROCK 4D (BLE Peripheral)]
   열기/닫기/정지 버튼                      CurtainController → (모터 Phase M)
   상태 표시(notify)                        BlueZ + Python BLE 서버
```

## 구성
- **보드 측 — BLE 주변장치(GATT 서버)**: BlueZ(이미 보드에 5.x) + Python BLE peripheral 라이브러리
  (`bless` 또는 `bluezero`). 커튼 제어 서비스를 advertise → 앱이 연결.
- **앱 측 — Flutter(BLE Central)**: `flutter_blue_plus` 로 스캔/연결/명령 전송/상태 수신.
  Material 3 기본 지원.
- 앱 코드는 별도 디렉토리/저장소: `app_flutter/` (또는 `kiy0518/ai_curtain_app`).

## GATT 규격 (초안 — 확정 필요)
- **Curtain Service** UUID: `a1c0de00-0000-1000-8000-00805f9b34fb` (예시, 확정 시 교체)
- **Command** (Write, Write-No-Response): `a1c0de01-...`
  - 페이로드(ASCII): `OPEN` / `CLOSE` / `STOP` (추후 위치 지원 시 `POS:NN`)
- **Status** (Read + Notify): `a1c0de02-...`
  - 페이로드(JSON 또는 compact): `{"state":"OPEN","gesture":"STOP","motor":false}`
  - 상태 변경 시 notify 로 앱에 푸시.
- **Info**(선택, Read): 펌웨어/디바이스 이름/버전.

## 앱 화면 (Material 3)
- [ ] 디바이스 스캔/연결(자동 재연결, 본딩 기억)
- [ ] 큰 버튼 3개: 열기 / 정지 / 닫기
- [ ] 현재 커튼 상태 + 최근 제스처(notify 실시간)
- [ ] (선택) 스케줄 보기/편집 — 단, 스케줄은 웹 API 가 풍부하므로 링크 또는 BLE 확장
- [ ] 연결 끊김/오프라인 표시

## 보드 측 작업
- [ ] BLE peripheral 라이브러리 선정(`bless` 권장) + BlueZ 광고 동작 확인
- [ ] `src/ble_server.py`: 서비스/특성 정의 + Command 수신 → `CurtainController.command(..., 'ble')`
- [ ] Status notify(상태머신/제스처 변경 시 push)
- [ ] `app.py`(또는 별도 프로세스)와 컨트롤러 공유 — 단일 컨트롤러 인스턴스 접근 방식 결정
      (같은 프로세스 스레드로 통합 vs 별도 서비스 + IPC)
- [ ] 부팅 자동실행(systemd) — 대시보드와 함께 또는 별도 유닛

## 앱 측 작업
- [ ] Flutter 프로젝트 생성(`app_flutter/`), `flutter_blue_plus` 연동
- [ ] 스캔/연결/명령/notify 구현
- [ ] Material 3 UI(버튼/상태), 다크/라이트
- [ ] Android/iOS 권한(블루투스/위치) 처리
- [ ] 빌드/배포(APK; iOS 는 별도)

## 보안 / 신뢰성
- BLE **본딩(페어링)** 필수 — 인증된 폰만 제어. 가능하면 명령 특성에 간단 키/페어링 PIN.
- 연결 끊김 시 앱 재연결, 보드는 다중 클라이언트/단일 제어 정책 정의.
- 모터 연동(Phase M) 전까지 명령은 컨트롤러 placeholder 상태만 변경.

## 오픈 이슈 / 결정 필요
- BLE 서버 라이브러리(`bless` vs `bluezero` vs 직접 D-Bus) 및 BlueZ 버전 호환.
- 앱↔보드 인증 방식(BLE 본딩만 vs 추가 토큰).
- 앱 코드 위치(이 저장소 하위 `app_flutter/` vs 별도 저장소).
- 스케줄/관리 기능은 BLE 로 확장할지, 웹 대시보드로 위임할지.
- iOS 배포(개발자 계정) 필요 여부 — 우선 Android APK.

## 의존성
- 핵심 제어(`CurtainController`)는 이미 존재 — BLE 는 또 하나의 입력 소스(제스처/대시보드/스케줄/BLE).
- 실제 구동은 모터(Phase M) 이후 의미. UI/연결/프로토콜은 먼저 구축 가능.
