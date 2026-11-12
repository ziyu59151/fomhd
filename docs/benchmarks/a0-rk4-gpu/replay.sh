#!/usr/bin/env bash
# Run both A0 sources sequentially inside the same single-GPU Slurm allocation.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    printf 'Usage: %s ORIGINAL_CHECKOUT OPTIMIZED_CHECKOUT NEW_OUTPUT [PYTHON]\n' "$0" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Run inside your own single-GPU Slurm allocation}"

packet=$(cd -- "$(dirname -- "$0")" && pwd)
original=$(realpath -- "$1")
optimized=$(realpath -- "$2")
python=$(command -v "${4:-python3}")
python=$(cd -- "$(dirname -- "$python")" && pwd)/$(basename -- "$python")
for source in "$original" "$optimized"; do
    test -f "$source/src/fomhd/__main__.py"
done
mkdir -- "$3"
run=$(realpath -- "$3")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH="$run/warp-cache"
gpu_filter=()
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    gpu_filter=(-i "$SLURM_JOB_GPUS")
fi
nvidia-smi "${gpu_filter[@]}" --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader > "$run/device.csv"

measure() {
    local label=$1 source=$2 grid=$3 repetition=$4
    local output="$run/$label-g$grid-r$repetition" pid watcher status
    (
        cd -- "$source"
        export PYTHONPATH="$source/src:$source"
        exec "$python" "$packet/measure.py" "$packet/inputs/solovev-g$grid/case.toml" \
            "$output" "$output.jsonl"
    ) > "$output.log" 2>&1 &
    pid=$!
    (
        while kill -0 "$pid" 2>/dev/null; do
            nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits |
                awk -v pid="$pid" -v time="$(date +%s.%N)" '$1==pid"," {print time "," $1 $2; fflush()}'
            sleep 0.1
        done
    ) > "$output-device.csv" &
    watcher=$!
    if wait "$pid"; then status=0; else status=$?; fi
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\n' "$label" "$grid" "$repetition" "$status" >> "$run/exits.tsv"
    if [[ "$status" != 0 ]]; then
        cat "$output.log" >&2
        return "$status"
    fi
}
measure original "$original" 12 1
measure optimized "$optimized" 12 1
measure original "$original" 12 2
measure optimized "$optimized" 12 2
measure optimized "$optimized" 24 1
