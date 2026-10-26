#!/usr/bin/env bash
# Mirror this repo's code to the GPU workstation, or pull what was generated there.
#   scripts/gpu_sync.sh push       # local code -> $GPU_WORKDIR/repo (never .env/.ssh/.git/outputs/artifacts,
#                                  # and never the files generated there, listed under pull)
#   scripts/gpu_sync.sh pull       # generated metadata -> local: data_pipeline/*.json, tokenizer/,
#                                  # training/*.json, eval/evaluation_results.json, eval/results/,
#                                  # release/*.json (push and access-verification reports)
#                                  # (code and configs are authored locally and only ever pushed)
#   scripts/gpu_sync.sh artifacts  # release export + generated audio -> local git-ignored artifacts/
# Uses the git-ignored .ssh/gpu_ssh.sh wrapper (password via SSH_ASKPASS from .env).
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=".ssh/gpu_ssh.sh"
REMOTE="gpu:vaani_pocket_tts_work/repo/"
case "${1:-}" in
  push)
    rsync -a -e "$SSH" \
      --exclude '.env' --exclude '.ssh/' --exclude '.git/' --exclude 'outputs/' --exclude 'artifacts/' \
      --exclude 'gpu_state_log.md' --exclude 'gpu_state_snapshots/' --exclude 'HANDOFF*.md' \
      --exclude '__pycache__/' --exclude '*.pyc' --exclude '.venv/' --exclude '.pytest_cache/' --exclude '.test_deps/' \
      --exclude 'data_pipeline/*.json' --exclude 'tokenizer/' --exclude 'training/*.json' \
      --exclude 'eval/evaluation_results.json' --exclude 'eval/results/' --exclude 'release/*.json' \
      ./ "$REMOTE"
    ;;
  pull)
    rsync -a -e "$SSH" --prune-empty-dirs \
      --include '*/' \
      --include 'data_pipeline/*.json' --include 'tokenizer/***' --include 'training/*.json' \
      --include 'eval/evaluation_results.json' --include 'eval/results/***' --include 'release/*.json' \
      --exclude '*' \
      "$REMOTE" ./
    ;;
  artifacts)
    mkdir -p artifacts
    rsync -a -e "$SSH" gpu:vaani_pocket_tts_work/release/ artifacts/release/
    rsync -a -e "$SSH" gpu:vaani_pocket_tts_work/eval_audio/ artifacts/eval_audio/
    ;;
  *) echo "usage: $0 push|pull|artifacts" >&2; exit 2 ;;
esac
