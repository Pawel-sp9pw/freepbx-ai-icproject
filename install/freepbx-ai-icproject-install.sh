#!/usr/bin/env bash

source /dev/stdin <<<"$FUNCTIONS_FILE_PATH"

color
verb_ip6
catch_errors
setting_up_container
network_check
update_os

msg_info "Installing dependencies"
$STD apt-get install -y \
  ca-certificates \
  curl \
  git \
  caddy \
  ffmpeg \
  build-essential \
  python3 \
  python3-venv \
  python3-pip \
  wireguard-tools
msg_ok "Installed dependencies"

msg_info "Installing FreePBX AI project"
REPO_URL="${REPO_URL:-https://github.com/Pawel-sp9pw/freepbx-ai-icproject.git}"
if [[ -d /opt/freepbx-ai-icproject ]]; then
  rm -rf /opt/freepbx-ai-icproject
fi
git clone "$REPO_URL" /opt/freepbx-ai-icproject
cd /opt/freepbx-ai-icproject
bash scripts/install-local.sh
msg_ok "Installed FreePBX AI project"

motd_ssh
customize
cleanup_lxc
