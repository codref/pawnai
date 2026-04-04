#!/usr/bin/env bash
# Install a single PawnAI systemd user service.
# Paths are resolved from the project root at install time — no hardcoding.
#
# Usage:
#   bash systemd/install.sh <service>
#
# Services: pawn-agent  pawn-diarize  litellm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

# ---------------------------------------------------------------------------
# Resolve venv binary directory
# ---------------------------------------------------------------------------
if [[ -d "$PROJECT_DIR/.venv/bin" ]]; then
    BIN_DIR="$PROJECT_DIR/.venv/bin"
else
    BIN_DIR="$(dirname "$(command -v pawn-agent 2>/dev/null || echo "/usr/local/bin/pawn-agent")")"
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 <service>"
    echo "  service: pawn-agent | pawn-diarize | litellm | litellm-docker"
    exit 1
}

[[ $# -eq 1 ]] || usage
SERVICE="$1"
[[ "$SERVICE" =~ ^(pawn-agent|pawn-diarize|litellm|litellm-docker)$ ]] || { echo "Unknown service: $SERVICE"; usage; }

# ---------------------------------------------------------------------------
# Substitute placeholders and install
# ---------------------------------------------------------------------------
TEMPLATE="$SCRIPT_DIR/$SERVICE.service"
[[ -f "$TEMPLATE" ]] || { echo "Template not found: $TEMPLATE"; exit 1; }

mkdir -p "$SYSTEMD_USER_DIR"
UNIT_FILE="$SYSTEMD_USER_DIR/$SERVICE.service"

sed \
    -e "s|{{PROJECT_DIR}}|$PROJECT_DIR|g" \
    -e "s|{{BIN_DIR}}|$BIN_DIR|g" \
    "$TEMPLATE" > "$UNIT_FILE"

echo "Installed $UNIT_FILE"
echo ""
cat "$UNIT_FILE"

# ---------------------------------------------------------------------------
# Enable and start
# ---------------------------------------------------------------------------
echo ""
systemctl --user daemon-reload
systemctl --user enable "$SERVICE"
systemctl --user restart "$SERVICE"

echo ""
systemctl --user status "$SERVICE" --no-pager -l
