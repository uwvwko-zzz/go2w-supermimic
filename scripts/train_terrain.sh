#!/usr/bin/env bash
# Go2W v7, fresh training by default; later flags override this preset.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${GO2W_PYTHON:-python}"
exec "$PYTHON_BIN" scripts/train.py \
  --terrain --terrain-revision 7 --command-curriculum \
  --num-envs 4096 --iterations 40000 --num-steps-per-env 1 \
  --wm-batch 2048 --wm-window 8 --wm-updates 4 \
  --imag-batch 8192 --horizon 32 --critic-epochs 5 --critic-minibatches 4 \
  --prioritized-replay --compile-imagination \
  --warmup 300 --actor-ramp-iterations 1000 --actor-hold-iterations 20000 \
  --lr-decay-iterations 20000 --lr-final-frac .2 "$@"
