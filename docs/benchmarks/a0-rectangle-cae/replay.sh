#!/usr/bin/env bash
# Replay one exact case inside a personal single-GPU Slurm allocation.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
    printf 'Usage: %s SOLVER_CHECKOUT CASE NEW_CASE_DIRECTORY [PYTHON]\n' "$0" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Run inside your own single-GPU Slurm allocation}"
packet=$(cd -- "$(dirname -- "$0")" && pwd)
source=$(realpath -- "$1")
case_name=$2
python=$(command -v "${4:-python3}")
python=$(cd -- "$(dirname -- "$python")" && pwd)/$(basename -- "$python")
case "$case_name" in
    narrow-g12|narrow-g16|quiet-g16|half-g16|wide-g20)
        expected=803b332ce45750b8c6e759f7405ba76ed87ee213
        ;;
    cold-hall-g8)
        expected=bc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369
        ;;
    *) printf 'Unknown case: %s\n' "$case_name" >&2; exit 2 ;;
esac
test "$(git -C "$source" rev-parse HEAD)" = "$expected"
git -C "$source" diff --quiet HEAD -- src/fomhd pyproject.toml uv.lock
mkdir -- "$3"
case_dir=$(realpath -- "$3")
cp -- "$packet/inputs/$case_name/"* "$case_dir/"
cd -- "$source"
export PYTHONPATH="$source/src:$source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export FOMHD_RECEIPT="$case_dir/timing.jsonl"
exec "$python" "$packet/instrument.py" run "$case_dir/case.toml"
