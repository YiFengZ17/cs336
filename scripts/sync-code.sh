#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/storage-common.sh"
DRY=false
if [[ "${1:-}" == --dry-run ]]; then OPTIONS+=(--dry-run); DRY=true; shift; fi
[[ $# == 0 ]] || { echo 'Usage: bash scripts/sync-code.sh [--dry-run]'; exit 2; }
ssh -o BatchMode=yes "$REMOTE" 'python3 -' <<'PY'
from pathlib import Path
import sys
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=p.read_bytes().decode().split('\0')
    except (OSError,UnicodeError): continue
    if any(Path(a).name in {'train.py','train_experiment.py','run_suite.py'} or a=='cs336_basics.part4_training_loop.train' for a in args):
        print('Training command detected; wait until it finishes before syncing.',file=sys.stderr)
        sys.exit(1)
PY
if ! $DRY; then python3 "$ROOT/scripts/write_sync_manifest.py"; fi
rsync "${OPTIONS[@]}" --backup --backup-dir="$REMOTE_RUNS/_maintenance/sync-$STAMP" \
  --exclude='.git' --exclude='.venv/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.vscode/' \
  --exclude='/data' --exclude='/checkpoints' --exclude='/logs' \
  --exclude='/a1-experiments/data/' --exclude='/a1-experiments/checkpoints' \
  --exclude='/a1-experiments/logs' --exclude='/a1-experiments/results' \
  --exclude='/a1-experiments/plans/' --exclude='/a1-experiments/profiles.local.json' \
  --exclude='/a1-experiments/profiles.server.json' \
  "$ROOT/" "$REMOTE:$REMOTE_CODE/"
