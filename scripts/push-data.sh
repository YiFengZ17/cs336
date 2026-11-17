#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/storage-common.sh"
[[ $# -ge 1 ]] || { echo 'Usage: bash scripts/push-data.sh RELATIVE_DATASET [--dry-run]'; exit 2; }
REL="$1"; shift; validate_relative "$REL"
if [[ "${1:-}" == --dry-run ]]; then OPTIONS+=(--dry-run); shift; fi
[[ $# == 0 && -d "$DATA_ROOT/$REL" ]] || { echo 'Invalid arguments or missing dataset'; exit 2; }
# Checksum transfer; previous destination files are retained in a dated backup.
OPTIONS+=(--mkpath)
rsync "${OPTIONS[@]}" --backup --backup-dir="$REMOTE_RUNS/_maintenance/data-$STAMP" \
  "$DATA_ROOT/$REL/" "$REMOTE:$REMOTE_DATA/$REL/"
