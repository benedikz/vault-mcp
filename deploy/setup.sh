#!/usr/bin/env bash
# Provision Brain Vault MCP on Ubuntu Server 24.04 (aarch64 / Raspberry Pi).
# Idempotent: safe to re-run. Run with sudo from the cloned repo:
#     sudo bash /opt/vault-mcp/deploy/setup.sh
#
# It installs OS deps, creates the `vault` service user and data dirs, builds
# the venv, and installs the package. It does NOT initialise the vault or write
# secrets — do that interactively afterwards (see README "Initialize").
set -euo pipefail

APP_DIR=/opt/vault-mcp
DATA_DIR=/var/lib/vault-mcp
ETC_DIR=/etc/vault-mcp
SVC_USER=vault

if [[ $EUID -ne 0 ]]; then echo "run with sudo" >&2; exit 1; fi

echo ">> apt dependencies"
apt-get update -y
apt-get install -y --no-install-recommends \
  python3-venv python3-dev build-essential pkg-config \
  libsqlcipher-dev libssl-dev git ca-certificates curl

echo ">> service user: ${SVC_USER}"
if ! id -u "${SVC_USER}" >/dev/null 2>&1; then
  useradd --system --home-dir "${DATA_DIR}" --shell /usr/sbin/nologin "${SVC_USER}"
fi

echo ">> directories"
install -d -m 0700 -o "${SVC_USER}" -g "${SVC_USER}" "${DATA_DIR}"
install -d -m 0700 -o "${SVC_USER}" -g "${SVC_USER}" "${DATA_DIR}/models" "${DATA_DIR}/hf"
install -d -m 0750 -o root -g "${SVC_USER}" "${ETC_DIR}"

echo ">> python venv + package"
if [[ ! -x "${APP_DIR}/.venv/bin/python" ]]; then
  python3 -m venv "${APP_DIR}/.venv"
fi
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -e "${APP_DIR}"

echo ">> systemd unit"
cp "${APP_DIR}/deploy/vault-mcp.service" /etc/systemd/system/vault-mcp.service
systemctl daemon-reload

cat <<EOF

Done. Next (see README "Initialize"):
  1. sudo -u ${SVC_USER} HOME=${DATA_DIR} HF_HOME=${DATA_DIR}/hf \\
       VAULT_DB_PATH=${DATA_DIR}/brain.db VAULT_CACHE=${DATA_DIR}/models \\
       VAULT_MODEL=intfloat/multilingual-e5-small \\
       ${APP_DIR}/.venv/bin/python ${APP_DIR}/bin/init.py
     -> copy the printed key into ${ETC_DIR}/secrets.env
  2. Write ${ETC_DIR}/secrets.env (chmod 640, root:${SVC_USER}):
       VAULT_KEY=<key from init>
       VAULT_TOKEN=<openssl rand -hex 24>
  3. sudo systemctl enable --now vault-mcp
EOF
