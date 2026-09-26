#!/usr/bin/env bash

# Community Scripts' build.func normally downloads application installers only
# from community-scripts/ProxmoxVE. This project is external, so patch the two
# installer download URLs while keeping the rest of the upstream LXC builder.
source <(
  curl -fsSL https://raw.githubusercontent.com/community-scripts/ProxmoxVE/main/misc/build.func |
    sed 's#https://raw.githubusercontent.com/community-scripts/ProxmoxVE/main/install/${var_install}.sh#https://raw.githubusercontent.com/Pawel-sp9pw/freepbx-ai-icproject/main/install/${var_install}.sh#g'
)

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

# Do not let variables() derive this from the human-readable APP name.
# Our installer is: install/freepbx-ai-icproject-install.sh
NSAPP="freepbx-ai-icproject"
var_install="${NSAPP}-install"

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
