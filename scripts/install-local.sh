#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/freepbx-ai-icproject"
VENV="/opt/freepbx-ai/.venv"

apt-get update
apt-get install -y \
  ca-certificates curl git caddy ffmpeg build-essential python3 python3-venv python3-pip wireguard-tools

mkdir -p /opt/freepbx-ai /var/lib/freepbx-ai/piper /etc/freepbx-ai

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel setuptools
"$VENV/bin/pip" install -r "$APP_DIR/requirements.txt"
"$VENV/bin/pip" install "piper-tts[http]"

# Ollama
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
systemctl enable --now ollama

# LLM
ollama pull qwen3:4b || true

# Piper PL
cd /var/lib/freepbx-ai/piper
"$VENV/bin/python" -m piper.download_voices pl_PL-mc_speech-medium

cp "$APP_DIR/systemd/freepbx-ai.service" /etc/systemd/system/freepbx-ai.service
cp "$APP_DIR/systemd/piper-ai.service" /etc/systemd/system/piper-ai.service
if [[ ! -f /etc/freepbx-ai/admin-password ]]; then
  openssl rand -base64 24 > /etc/freepbx-ai/admin-password
  chmod 600 /etc/freepbx-ai/admin-password
fi

cat >/etc/caddy/Caddyfile <<'EOF'
:8080 {
    reverse_proxy 127.0.0.1:8000
}
EOF

systemctl daemon-reload
systemctl enable --now piper-ai
systemctl enable --now freepbx-ai
systemctl enable --now caddy
systemctl restart caddy

echo
echo "=============================================================="
echo " FreePBX AI -> IC Project zainstalowany"
echo " Panel: http://$(hostname -I | awk '{print $1}'):8080"
echo " AudioSocket: $(hostname -I | awk '{print $1}'):9019"
echo "=============================================================="
