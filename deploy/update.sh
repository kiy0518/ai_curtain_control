#!/usr/bin/env bash
# OTA-style update: pull latest code and restart the service (if installed).
set -e
cd "$(dirname "$0")/.."
git pull --ff-only
if systemctl is-enabled ai-curtain >/dev/null 2>&1; then
  sudo systemctl restart ai-curtain
  echo "updated + restarted (systemd)."
else
  echo "updated. (systemd 미설치 — 앱을 수동 재시작하세요)"
fi
