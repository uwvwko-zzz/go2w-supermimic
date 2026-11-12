#!/usr/bin/env bash
# Independent vision checkpoint launcher; later arguments override defaults.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$project_dir/scripts/play_vision_gui.sh" \
  --checkpoint "$project_dir/logs/go2w_vision_30000/model_30000.pt" \
  --device cuda:0 --port 8766 --terrain-kind platform --terrain-level 1 "$@"
