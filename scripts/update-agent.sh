#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="/opt/freepbx-ai-icproject"
VENV="/opt/freepbx-ai/.venv"
STATE="/var/lib/freepbx-ai/update.state"
LOG="/var/lib/freepbx-ai/update.log"

mkdir -p /var/lib/freepbx-ai
: > "$LOG"
echo "running" > "$STATE"

exec > >(tee -a "$LOG") 2>&1

finish() {
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "success" > "$STATE"
    echo
    echo "Aktualizacja zakończona pomyślnie."
  else
    echo "error:$rc" > "$STATE"
    echo
    echo "Aktualizacja zakończona błędem (kod $rc)."
  fi
}
trap finish EXIT

echo "=== FreePBX AI update ==="
date -Is

cd "$APP_DIR"

echo
echo "[1/6] Pobieranie zmian z GitHub..."
git fetch origin main
git pull --ff-only origin main

echo
echo "[2/6] Aktualizacja pakietów Python..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r requirements.txt
"$VENV/bin/pip" install "piper-tts[http]"

echo
echo "[3/6] Aktualizacja jednostek systemd..."
install -m 0644 systemd/freepbx-ai.service /etc/systemd/system/freepbx-ai.service
install -m 0644 systemd/piper-ai.service /etc/systemd/system/piper-ai.service
systemctl daemon-reload

echo
echo "[4/6] Restart Piper..."
systemctl restart piper-ai

echo
echo "[5/6] Restart agenta i panelu..."
systemctl restart freepbx-ai

echo
echo "[6/6] Weryfikacja..."
sleep 2
systemctl is-active --quiet freepbx-ai
systemctl is-active --quiet piper-ai
echo "Commit: $(git rev-parse --short HEAD)"
echo "Agent: $(systemctl is-active freepbx-ai)"
echo "Piper: $(systemctl is-active piper-ai)"
