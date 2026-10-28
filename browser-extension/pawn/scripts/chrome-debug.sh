#!/usr/bin/env bash
# Relaunch Google Chrome with a remote-debugging port so agents can CDP-attach.
# Usage:
#   1. Quit Chrome completely (check the system tray / `pkill chrome` if needed).
#   2. ./browser-extension/pawn/scripts/chrome-debug.sh
#   3. Open Teams + reload the unpacked Pawn extension.
#   4. Tell the agent: debugging is on port 9222.
set -euo pipefail
PORT="${1:-9222}"
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port ${PORT} already listening — Chrome may already be in debug mode."
  exit 0
fi
if pgrep -x chrome >/dev/null 2>&1 || pgrep -f '/opt/google/chrome/chrome' >/dev/null 2>&1; then
  echo "Chrome is still running. Quit it fully, then re-run this script."
  echo "  pgrep -af chrome | head"
  exit 1
fi
PROFILE_DIR="${HOME}/.config/google-chrome-pawn-debug"
mkdir -p "${PROFILE_DIR}"
echo "Starting Chrome with --remote-debugging-port=${PORT}"
echo "Profile: ${PROFILE_DIR} (separate from your daily profile — sign into Teams once)."
exec /opt/google/chrome/chrome \
  --remote-debugging-port="${PORT}" \
  --remote-allow-origins=* \
  --user-data-dir="${PROFILE_DIR}" \
  "https://teams.cloud.microsoft/" 
