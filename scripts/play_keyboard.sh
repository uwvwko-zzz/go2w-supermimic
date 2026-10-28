#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  keyboard_python="$project_dir/.venv/bin/python"
else
  keyboard_python=python
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$keyboard_python" scripts/play_keyboard.py "$@"
