#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -n "${GO2W_VISION_PYTHON:-}" ]]; then
  vision_python="$GO2W_VISION_PYTHON"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
  vision_python="$project_dir/.venv/bin/python"
else
  vision_python=python
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$vision_python" scripts/play_vision_gui.py "$@"
