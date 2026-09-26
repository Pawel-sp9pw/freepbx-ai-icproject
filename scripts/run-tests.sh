#!/usr/bin/env bash
set -euo pipefail

cd /opt/freepbx-ai-icproject 2>/dev/null || cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/opt/freepbx-ai/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python3
fi

exec "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
