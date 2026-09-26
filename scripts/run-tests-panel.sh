#!/usr/bin/env bash
set -Eeuo pipefail

STATE="/var/lib/freepbx-ai/tests.state"
LOG="/var/lib/freepbx-ai/tests.log"
APP_DIR="/opt/freepbx-ai-icproject"

mkdir -p /var/lib/freepbx-ai
: > "$LOG"
echo "running" > "$STATE"

exec > >(tee -a "$LOG") 2>&1

finish() {
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "success" > "$STATE"
    echo
    echo "Wszystkie testy zakończone pomyślnie."
  else
    echo "error:$rc" > "$STATE"
    echo
    echo "Testy zakończone błędem (kod $rc)."
  fi
}
trap finish EXIT

echo "=== FreePBX AI regression tests ==="
date -Is
echo

cd "$APP_DIR"
/bin/bash "$APP_DIR/scripts/run-tests.sh"
