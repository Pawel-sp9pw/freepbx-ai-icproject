#!/usr/bin/env bash
set -euo pipefail

CTID="${1:-}"

if [[ -z "$CTID" || ! "$CTID" =~ ^[0-9]+$ ]]; then
  echo "Użycie: bash scripts/repair-existing-lxc.sh <CTID>"
  echo "Przykład: bash scripts/repair-existing-lxc.sh 101"
  exit 1
fi

if ! command -v pct >/dev/null 2>&1; then
  echo "Ten skrypt uruchom na hoście Proxmox."
  exit 1
fi

echo "[1/4] Sprawdzanie CT $CTID"
pct status "$CTID"

echo "[2/4] Instalacja git/curl w kontenerze"
pct exec "$CTID" -- bash -lc 'apt-get update && apt-get install -y git curl ca-certificates'

echo "[3/4] Pobieranie projektu"
pct exec "$CTID" -- bash -lc '
  rm -rf /opt/freepbx-ai-icproject
  git clone https://github.com/Pawel-sp9pw/freepbx-ai-icproject.git /opt/freepbx-ai-icproject
'

echo "[4/4] Instalacja aplikacji"
pct exec "$CTID" -- bash -lc '
  cd /opt/freepbx-ai-icproject
  bash scripts/install-local.sh
'

IP="$(pct exec "$CTID" -- hostname -I | awk "{print \$1}")"

echo
echo "Naprawa zakończona."
echo "Panel: http://${IP}:8080"
echo "Sprawdzenie:"
echo "  pct exec $CTID -- systemctl status freepbx-ai --no-pager"
echo "  pct exec $CTID -- systemctl status caddy --no-pager"
echo "  pct exec $CTID -- ss -tlnp | grep -E ':8080 |:9019 '"
