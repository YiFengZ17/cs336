#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/storage-common.sh"
[[ $# -ge 1 ]] || { echo 'Usage: bash scripts/pull-results.sh RELATIVE_RUN [--with-checkpoints] [--dry-run]'; exit 2; }
REL="$1"; shift; validate_relative "$REL"
WITH_WEIGHTS=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) OPTIONS+=(--dry-run);;
    --with-checkpoints) WITH_WEIGHTS=true;;
    *) echo "Unknown option: $arg";exit 2;;
  esac
done
if ! $WITH_WEIGHTS; then OPTIONS+=(--exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' --exclude='*.bin'); fi
DEST="$RUN_ROOT/server-download/$REL"
mkdir -p "$DEST"
rsync "${OPTIONS[@]}" --backup --backup-dir="$RUN_ROOT/_maintenance/download-$STAMP" \
  "$REMOTE:$REMOTE_RUNS/$REL/" "$DEST/"
