#!/usr/bin/env bash
# AI Curtain Control — board setup. Run from the repo root: bash deploy/install.sh
set -e
cd "$(dirname "$0")/.."

echo "[1/3] OpenCV (GStreamer 포함) 설치 ..."
sudo apt-get update -qq
sudo apt-get install -y python3-opencv

echo "[2/3] cloudflared (원격 접속, 선택) ..."
if [ ! -x "$HOME/.local/bin/cloudflared" ]; then
  mkdir -p "$HOME/.local/bin"
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 \
    -o "$HOME/.local/bin/cloudflared" && chmod +x "$HOME/.local/bin/cloudflared"
fi

echo "[3/3] config.env ..."
[ -f config.env ] || cp config.env.example config.env

cat <<'EOF'

완료. 실행:
  python3 app.py                 # 대시보드 (기본 비번 admin)
부팅 자동실행(systemd):
  sudo cp deploy/ai-curtain.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now ai-curtain
EOF
