#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  gui_python="$project_dir/.venv/bin/python"
else
  gui_python=python
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$gui_python" scripts/play_gui.py "$@"
