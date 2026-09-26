#!/usr/bin/env bash
set -Eeuo pipefail

export HOME="${HOME:-/root}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$HOME" "$XDG_CACHE_HOME"

MODEL="${1:?Brak nazwy modelu}"
STATE="/var/lib/freepbx-ai/model-pull.state"
LOG="/var/lib/freepbx-ai/model-pull.log"
CURRENT="/var/lib/freepbx-ai/model-pull.model"

mkdir -p /var/lib/freepbx-ai
printf '%s\n' "$MODEL" > "$CURRENT"
printf '%s\n' "running" > "$STATE"
: > "$LOG"

exec > >(tee -a "$LOG") 2>&1

finish() {
  rc=$?
  if [ "$rc" -eq 0 ]; then
    printf '%s\n' "success" > "$STATE"
  else
    printf '%s\n' "error:$rc" > "$STATE"
  fi
}
trap finish EXIT

echo "Pobieranie modelu: $MODEL"
date -Is
ollama pull "$MODEL"
echo "Model $MODEL pobrany."
