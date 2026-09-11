#!/usr/bin/env bash
# Install or update the collector. Safe to re-run: it changes what has drifted
# and leaves the rest, and it never touches the accumulated corpus.
#
#   sudo ./deploy/install.sh [/opt/btc-intel]
#
set -euo pipefail

PREFIX="${1:-/opt/btc-intel}"
SERVICE_USER="btc-intel"
STATE_ROOT="/var/lib/btc-intel"
UNIT_DIR="/etc/systemd/system"
ENV_DIR="/etc/btc-intel"
ENV_FILE="$ENV_DIR/collector.env"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ $EUID -eq 0 ]] || { echo "install.sh must run as root" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "no systemd on this host" >&2; exit 1; }

echo "==> service account"
if id -u "$SERVICE_USER" >/dev/null 2>&1; then
    echo "    $SERVICE_USER exists"
else
    # No login, no home, no shell: this account exists to own a state directory.
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    echo "    created $SERVICE_USER"
fi

echo "==> code at $PREFIX"
mkdir -p "$PREFIX"
if [[ "$SOURCE" != "$PREFIX" ]]; then
    # Deliberately no --delete. This must never be able to remove a state
    # directory an operator has placed inside the prefix; a deployment that can
    # eat its own corpus eventually does.
    rsync -a --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'btc-intel-state' "$SOURCE/" "$PREFIX/"
fi

echo "==> virtualenv"
if [[ ! -x "$PREFIX/.venv/bin/python" ]]; then
    python3 -m venv "$PREFIX/.venv"
fi
"$PREFIX/.venv/bin/python" -m pip install --quiet --upgrade pip
"$PREFIX/.venv/bin/python" -m pip install --quiet -r "$PREFIX/requirements-market-intelligence.txt"

echo "==> ownership"
# The code is read-only to the service account. Only the state directory is not.
chown -R root:root "$PREFIX"
chmod -R go-w "$PREFIX"

echo "==> configuration"
mkdir -p "$ENV_DIR"
if [[ -f "$ENV_FILE" ]]; then
    # An existing file may hold a real credential. Never overwrite it.
    echo "    $ENV_FILE exists, left alone"
else
    install -m 0640 -o root -g "$SERVICE_USER" "$PREFIX/deploy/collector.env.example" "$ENV_FILE"
    echo "    wrote $ENV_FILE from the example"
fi

# Record which commit is deployed, so a manifest can name the code that made it.
if SHA="$(git -C "$SOURCE" rev-parse HEAD 2>/dev/null)"; then
    sed -i '/^BTC_INTEL_SOURCE_SHA=/d' "$ENV_FILE"
    echo "BTC_INTEL_SOURCE_SHA=$SHA" >> "$ENV_FILE"
    echo "    deployed sha $SHA"
else
    echo "    not a git checkout; manifests will record no source sha" >&2
fi

echo "==> units"
for unit in "$PREFIX"/deploy/systemd/*.service "$PREFIX"/deploy/systemd/*.timer; do
    install -m 0644 -o root -g root "$unit" "$UNIT_DIR/$(basename "$unit")"
done
systemctl daemon-reload

echo "==> preflight"
# Prove the storage root is real and writable before enabling a timer against
# it. A collector that starts and cannot write is the exact failure this whole
# deployment exists to avoid: it looks identical to a quiet news week.
if ! sudo -u "$SERVICE_USER" env BTC_INTEL_STATE_ROOT="$STATE_ROOT" PYTHONPATH="$PREFIX" "$PREFIX/.venv/bin/python" "$PREFIX/btc-intel.py" ops-paths; then
    echo "storage preflight failed; timers NOT enabled" >&2
    exit 1
fi

echo "==> configuration check"
# The same check `ops-config-check` gives an operator: the profile loads, the
# contact is set, no retired feed is enabled, the state root is usable. Without
# a contact every cycle would fail on its first line, so no timer is enabled
# until there is one. The address is read from the env file here and never
# printed.
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
if ! sudo -u "$SERVICE_USER" env BTC_INTEL_CONTACT="${BTC_INTEL_CONTACT:-}" BTC_INTEL_STATE_ROOT="$STATE_ROOT" PYTHONPATH="$PREFIX" \
        "$PREFIX/.venv/bin/python" "$PREFIX/btc-intel.py" ops-config-check --profile "$PREFIX/deploy/collection-profile.json"; then
    echo "configuration not deployable; set BTC_INTEL_CONTACT in $ENV_FILE and re-run. timers NOT enabled" >&2
    exit 1
fi

echo "==> timers"
systemctl enable --now btc-intel-collect.timer
systemctl enable --now btc-intel-verify.timer
systemctl enable --now btc-intel-backup.timer
systemctl enable --now btc-intel-watch.timer

echo
echo "installed. next runs:"
systemctl list-timers --no-pager 'btc-intel-*' || true
echo
echo "run one cycle now:  systemctl start btc-intel-collect.service"
echo "watch it:           journalctl -u btc-intel-collect.service -f"
