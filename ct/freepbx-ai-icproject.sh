#!/usr/bin/env bash
source <(curl -fsSL https://raw.githubusercontent.com/community-scripts/ProxmoxVE/main/misc/build.func)

# Copyright (c) 2026
# Inspired by community-scripts.org layout.
# This is an independent project, not an official Community Scripts entry.

APP="FreePBX AI ICProject"
var_tags="ai;freepbx;asterisk;icproject"
var_cpu="4"
var_ram="8192"
var_disk="24"
var_os="debian"
var_version="13"
var_unprivileged="1"

header_info "$APP"
base_settings
variables
color
catch_errors

function update_script() {
  header_info
  check_container_storage
  check_container_resources

  if [[ ! -d /opt/freepbx-ai-icproject ]]; then
    msg_error "No ${APP} installation found!"
    exit
  fi

  msg_info "Updating ${APP}"
  cd /opt/freepbx-ai-icproject
  git pull
  /opt/freepbx-ai/.venv/bin/pip install -r requirements.txt
  systemctl restart freepbx-ai piper-ai
  msg_ok "Updated ${APP}"
  exit
}

start
build_container
description

msg_ok "Completed Successfully!"
echo -e "${APP} should be reachable at:"
echo -e "  http://${IP}:8080"
