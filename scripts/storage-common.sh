#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${CS336_REMOTE:-mentor-gpu}"
DATA_ROOT="${CS336_DATA_ROOT:-$HOME/datasets/CS336}"
RUN_ROOT="${CS336_RUN_ROOT:-$HOME/experiments/CS336}"
REMOTE_CODE=/home/zyf/projects/CS336
REMOTE_DATA=/home/zyf/datasets/CS336
REMOTE_RUNS=/home/zyf/experiments/CS336
RSYNC_REMOTE=/home/zyf/.local/bin/rsync
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OPTIONS=(-a --checksum --itemize-changes --human-readable --progress
  --rsync-path="$RSYNC_REMOTE" -e 'ssh -o BatchMode=yes -o ConnectTimeout=10')
validate_relative() {
  [[ -n "$1" && "$1" != /* && "$1" != -* && "/$1/" != *"/../"* && "$1" =~ ^[A-Za-z0-9_./-]+$ ]] || {
    echo 'Use a relative path with letters, digits, underscores, hyphens and slashes.' >&2; exit 2;
  }
}
