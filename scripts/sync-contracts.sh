#!/usr/bin/env bash
# Copy the voice assistant contract (golden SSE transcripts + shared utterances)
# into the client repos. The copy in this repo is canonical; never edit the copies.
#
#   scripts/sync-contracts.sh [LINUX_REPO] [ANDROID_REPO]
#   scripts/sync-contracts.sh --check   # exit 1 when a copy differs
#
# Defaults: ../pawnai-assistant and ../pawnai-assistant-android next to this repo.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/contracts"
check=0
if [[ "${1:-}" == "--check" ]]; then
  check=1
  shift
fi
linux="${1:-$here/../pawnai-assistant}"
android="${2:-$here/../pawnai-assistant-android}"

targets=()
[[ -d "$linux" ]] && targets+=("$linux/tests/contracts")
[[ -d "$android" ]] && targets+=("$android/skill-testkit/src/main/resources/contracts")

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "no client repo found (looked for $linux and $android)" >&2
  exit 1
fi

status=0
for dst in "${targets[@]}"; do
  if [[ $check -eq 1 ]]; then
    if ! diff -r -q "$src" "$dst" >/dev/null 2>&1; then
      echo "out of date: $dst" >&2
      status=1
    else
      echo "ok: $dst"
    fi
  else
    mkdir -p "$dst"
    rsync -a --delete "$src/" "$dst/"
    echo "synced: $dst"
  fi
done
exit $status
