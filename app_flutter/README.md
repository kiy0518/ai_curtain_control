# AI Curtain — Flutter BLE 리모컨 앱

보드(ROCK 4D)의 BLE 주변장치에 연결해 커튼을 **열기/정지/닫기** 하고 상태를 받는 안드로이드 앱.
(보드 측: `src/ble_server.py`, 서비스 `c0de0000-…`)

## 빌드 (PC에서, Flutter SDK 필요 — 보드 아님)

```bash
# 1) 이 폴더에서 Flutter 프로젝트 골격 생성(android/ios 플랫폼 파일)
cd app_flutter
flutter create .            # 기존 pubspec.yaml/lib/main.dart 유지됨
flutter pub get

# 2) Android 권한 추가 — android/app/src/main/AndroidManifest.xml 의
#    <manifest> 안, <application> 위에 아래 추가 (manifest_permissions.xml 참고)

# 3) 빌드
flutter build apk --release
#  결과: build/app/outputs/flutter-apk/app-release.apk → 폰에 설치
```

## 사용
1. 보드에서 BLE 켜기: `python3 app.py --ble` (또는 config.env `CURTAIN_BLE=1`)
2. 폰에서 앱 실행 → **기기 연결** → "AI-Curtain" 자동 검색·연결
3. 열기/정지/닫기 버튼, 상단에 커튼 상태 실시간 표시
4. 보안: **BLE 본딩(페어링)** 으로 인증된 폰만 (OS 페어링)

## 참고
- 패키지: `flutter_blue_plus`
- Material 3 (`useMaterial3: true`)
- 실제 커튼 구동은 보드 모터 연동(Phase M) 이후. 그 전엔 상태 placeholder.
- iOS 는 추후(개발자 계정 필요). 우선 Android APK.
