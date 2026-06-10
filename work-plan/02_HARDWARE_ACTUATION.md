# 02. 커튼 물리 제어 (Motor Controller — Serial)

> Phase 1 · 목표버전 v0.2.0 · 상태: 미착수

제스처/스케줄로 결정된 동작을 **모터 컨트롤러**로 전달해 실제 커튼을 구동하는 계층.
연결은 **시리얼 통신(UART / USB-Serial)** 으로 한다 (보드 GPIO 직접 구동 X — 컨트롤러가 모터 전력단 담당).

```
[제스처/스케줄/대시보드] → control 로직 → 시리얼 드라이버 ──(UART)──► [모터 컨트롤러] → 모터/커튼
                                            ▲ 상태/ACK ◄───────────────┘
```

## 목표
- 열림/닫힘/정지/위치(%)/**속도** 명령을 시리얼로 송신.
- 컨트롤러 상태(위치·이동상태·끝단·오류) 수신·반영.
- 통신 프로토콜을 명확히 정의(양측 합의) + 재연결/타임아웃/오류 처리.

## 시리얼 연결 파라미터 (config.env)
| 항목 | 기본값(예) | 비고 |
|------|-----------|------|
| `MOTOR_SERIAL_PORT` | `/dev/ttyS3` 또는 `/dev/ttyUSB0` | 보드 UART / USB-시리얼 |
| `MOTOR_BAUDRATE` | `115200` | 8N1 |
| `MOTOR_TIMEOUT` | `0.3` s | 읽기 타임아웃 |
| `MOTOR_ACK_TIMEOUT` | `1.0` s | 명령 ACK 대기 |

## 통신 프로토콜 (제안 — v1)

ASCII 라인 기반, 줄 끝 `\n`. 사람이 읽기 쉽고 디버깅 용이. (필요 시 체크섬/바이너리로 확장)

### 프레임 형식
```
$<CMD>[,<arg1>[,<arg2>...]]*<CK>\n      # 체크섬 옵션(*CK 생략 가능, v1은 생략 허용)
```
- `$` 시작, `,` 인자 구분, `\n` 종료. `<CK>` = 앞 문자열 XOR(2자리 hex, 옵션).

### 호스트 → 컨트롤러 (명령)
| 명령 | 형식 | 의미 |
|------|------|------|
| 열기 | `$OPEN` | 완전 열림까지 이동 |
| 닫기 | `$CLOSE` | 완전 닫힘까지 이동 |
| 정지 | `$STOP` | 즉시 정지 |
| 위치 | `$POS,<0-100>` | 0=닫힘 … 100=열림 위치로 이동 |
| **속도** | `$SPEED,<0-100>` | 이동 속도(%) 설정 (또는 RPM) |
| 위치+속도 | `$MOVE,<pos>,<speed>` | 속도 지정하여 위치 이동 |
| 상태조회 | `$GET` | 현재 상태 요청 |
| 보정 | `$CAL` | 끝단 캘리브레이션 |
| 핑 | `$PING` | 생존 확인 |
| 핸드셰이크 | `$HELLO,<proto_ver>` | 접속 시 프로토콜 버전 교환 |

### 컨트롤러 → 호스트 (응답/이벤트)
| 응답 | 형식 | 의미 |
|------|------|------|
| 정상 | `#OK[,<data>]` | 명령 수락 |
| 오류 | `#ERR,<code>,<msg>` | 거부/실패 |
| 상태 | `#STA,pos=<0-100>,state=<IDLE\|OPENING\|CLOSING\|STOPPED\|ERROR>,speed=<0-100>,limit=<none\|open\|close>` | `$GET` 응답 |
| 이벤트 | `#EVT,<type>[,<data>]` | 비동기 알림(끝단도달/장애물/끼임 등) |
| 핑 | `#PONG` | 응답 |

### 오류 코드(예)
`E01` 미보정, `E02` 끝단 충돌, `E03` 과전류/끼임, `E04` 모터 정지 실패, `E09` 알수없음.

### 예시 시퀀스
```
$HELLO,1        →  #OK,proto=1,fw=1.2.0
$SPEED,60       →  #OK
$OPEN           →  #OK
                   #STA,pos=0,state=OPENING,speed=60,limit=none
                   #STA,pos=100,state=STOPPED,speed=60,limit=open
                   #EVT,reached_open
$GET            →  #STA,pos=100,state=IDLE,speed=60,limit=open
$STOP           →  #OK
```

## 설계 (보드측)
- `control/motor_serial.py`: pyserial 기반 송수신 + 수신 스레드(라인 파서) + ACK 매칭(요청/응답 큐). Azone_Gateway `registrator.py`의 큐 기반 UART 구조 참고.
- `control/curtain.py`: 상태머신(`#STA` 반영) + 고수준 API(`open/close/stop/set_position/set_speed`).
- 재연결: 포트 분리 감지 시 재오픈, `$PING/#PONG` 주기 헬스체크.
- 안전: ACK 타임아웃·`#ERR`·`#EVT(끼임)` 처리 → 상위에 전파, 위험 시 자동 `$STOP`.

## 작업
- [ ] 컨트롤러 측과 **프로토콜 v1 합의**(위 표 확정/수정)
- [ ] 시리얼 포트/배선 확인(보드 UART 핀 또는 USB-시리얼), `config.env` 항목 추가
- [ ] `motor_serial.py`(송수신·파서·ACK·재연결) 구현
- [ ] `curtain.py` 상태머신 + 속도 설정 반영
- [ ] 루프백/시뮬레이터(`test_motor_sim.py`)로 프로토콜 검증 (실모터 없이)
- [ ] 제스처 엔진 → 커튼 명령 연결(`serve.py`/control 통합)
- [ ] 끼임/끝단 이벤트 안전 처리

## 오픈 이슈 / 확인 필요
- 컨트롤러 **펌웨어가 이미 있는지**, 있다면 **기존 명령 규격**은? (있으면 그걸 따르고 위 제안은 폐기/조정)
- 속도 단위: **% vs RPM vs mm/s** 중 무엇?
- 위치 피드백 지원 여부(절대 위치% 가능한지, 아니면 개/폐만?)
- 보드 UART 포트 번호(`/dev/ttyS?`)와 전압 레벨(3.3V), USB-시리얼 사용 여부.
- 체크섬/프레이밍 필요 수준(노이즈 환경?).
