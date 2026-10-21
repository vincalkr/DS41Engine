#!/usr/bin/env bash
# install-power-tune.sh [uninstall] - once, with sudo: lets ./run.sh switch local/power-tune.sh on when the server
# starts and off when it ends, without asking for a password.
#
#   sudo local/install-power-tune.sh             install (or update after local/power-tune.sh changed)
#   sudo local/install-power-tune.sh uninstall
#
# It copies local/power-tune.sh to /usr/local/sbin/ds41e-power-tune (owned by root, not writable by the user: the
# rule below runs it as root, so a copy the user could edit would hand out root) and writes
# /etc/sudoers.d/ds41e-power-tune, which lets the user who ran sudo run exactly that file with "on", "off" or
# "show" and nothing else.  The sudoers file is checked with visudo before it is put in place.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run it with sudo: sudo $0 ${1:-}" >&2; exit 1; }
BIN=/usr/local/sbin/ds41e-power-tune
RULE=/etc/sudoers.d/ds41e-power-tune
if [ "${1:-}" = uninstall ]; then
    rm -f "$BIN" "$RULE"
    echo "removed $BIN and $RULE"
    exit 0
fi
U=${SUDO_USER:?run it with sudo from your own user, not from a root shell}
SRC="$(cd "$(dirname "$0")" && pwd)/power-tune.sh"
install -o root -g root -m 0755 "$SRC" "$BIN"
TMP=$(mktemp)
printf '%s ALL=(root) NOPASSWD: %s on, %s off, %s show\n' "$U" "$BIN" "$BIN" "$BIN" > "$TMP"
visudo -cf "$TMP" >/dev/null
install -o root -g root -m 0440 "$TMP" "$RULE"
rm -f "$TMP"
echo "installed $BIN (root:root 0755) and $RULE: $U may run 'sudo -n $BIN on|off|show'"
echo "./run.sh now switches it on at start and off when the server ends (DS41_POWER_TUNE=0 turns that off)"
