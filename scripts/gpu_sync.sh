#!/usr/bin/env bash
# Mirror this repo's code to the GPU workstation, or pull generated metadata back.
#   scripts/gpu_sync.sh push   # local code -> $GPU_WORKDIR/repo (never .env/.ssh/.git/outputs, and never
#                              # the artifacts generated there: data_pipeline/*.json, tokenizer/)
#   scripts/gpu_sync.sh pull   # artifacts generated there (data_pipeline/*.json, tokenizer/) -> local
#                              # (code and configs are authored locally and only ever pushed)
# Uses the git-ignored .ssh/gpu_ssh.sh wrapper (password via SSH_ASKPASS from .env).
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=".ssh/gpu_ssh.sh"
REMOTE="gpu:vaani_pocket_tts_work/repo/"
case "${1:-}" in
  push)
    rsync -a -e "$SSH" \
      --exclude '.env' --exclude '.ssh/' --exclude '.git/' --exclude 'outputs/' \
      --exclude 'gpu_state_log.md' --exclude 'gpu_state_snapshots/' --exclude 'HANDOFF*.md' \
      --exclude '__pycache__/' --exclude '*.pyc' --exclude '.venv/' \
      --exclude 'data_pipeline/*.json' --exclude 'tokenizer/' \
      ./ "$REMOTE"
    ;;
  pull)
    rsync -a -e "$SSH" --prune-empty-dirs \
      --include '*/' \
      --include 'data_pipeline/*.json' --include 'tokenizer/***' \
      --exclude '*' \
      "$REMOTE" ./
    ;;
  *) echo "usage: $0 push|pull" >&2; exit 2 ;;
esac
