#!/usr/bin/env bash
#SBATCH --job-name=fomhd-a0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Submit with explicit paths, for example:
# sbatch --export=ALL,FOMHD_PROJECT=/abs/frozen/checkout,FOMHD_PYTHON=/abs/frozen/checkout/.venv/bin/python,FOMHD_CASE=/abs/case.toml,FOMHD_OUTPUT=/abs/run jobs/a0.slurm
set -euo pipefail

: "${FOMHD_PROJECT:?Set FOMHD_PROJECT to the frozen Python/Warp checkout}"
: "${FOMHD_PYTHON:?Set FOMHD_PYTHON to the frozen checkout Python}"
: "${FOMHD_CASE:?Set FOMHD_CASE to the input case.toml}"
: "${FOMHD_OUTPUT:?Set FOMHD_OUTPUT to a new run directory}"
: "${SLURM_JOB_ID:?Run this launcher inside a Slurm allocation}"

if [[ "${FOMHD_PROJECT}" != /* || "${FOMHD_PYTHON}" != /* || "${FOMHD_CASE}" != /* || "${FOMHD_OUTPUT}" != /* ]]; then
  echo "FOMHD_PROJECT, FOMHD_PYTHON, FOMHD_CASE and FOMHD_OUTPUT must be absolute" >&2
  exit 2
fi

if [[ ! -d "${FOMHD_PROJECT}" || ! -f "${FOMHD_PROJECT}/pyproject.toml" || ! -f "${FOMHD_PROJECT}/uv.lock" || ! -d "${FOMHD_PROJECT}/src/fomhd" ]]; then
  echo "FOMHD_PROJECT is not a complete src-layout checkout: ${FOMHD_PROJECT}" >&2
  exit 2
fi
if [[ ! -x "${FOMHD_PYTHON}" ]]; then
  echo "FOMHD_PYTHON is not an executable: ${FOMHD_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${FOMHD_CASE}" ]]; then
  echo "FOMHD_CASE is not a regular file: ${FOMHD_CASE}" >&2
  exit 2
fi

output_parent=$(dirname -- "${FOMHD_OUTPUT}")
mkdir -p -- "${output_parent}"
launch_log="${FOMHD_OUTPUT}.launch.log"
project=$(realpath -- "${FOMHD_PROJECT}")
python=$(cd -- "$(dirname -- "${FOMHD_PYTHON}")" && pwd -P)/$(basename -- "${FOMHD_PYTHON}")
case_path=$(realpath -- "${FOMHD_CASE}")
expected_package="${project}/src/fomhd"
export FOMHD_PROJECT="${project}"
cd -- "${project}"
export PYTHONPATH="${project}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS=1
"${python}" - <<'PY'
import os
import pathlib
import sys

import fomhd

expected = pathlib.Path(os.environ["FOMHD_PROJECT"]).resolve() / "src" / "fomhd"
actual = pathlib.Path(fomhd.__file__).resolve().parent
if actual != expected:
    raise SystemExit(f"Python/Warp import mismatch: {actual} != {expected}")
PY
package_version=$(
  "${python}" - <<'PY'
import sys

import cupy
import warp

print(
    "cupy=" + cupy.__version__
    + " warp=" + warp.__version__
    + " python=" + sys.version.split()[0]
)
PY
)
{
  printf 'project=%s\n' "${project}"
  printf 'python=%s\n' "${python}"
  printf 'python_sha256=%s\n' "$(sha256sum -- "${python}" | awk '{print $1}')"
  printf 'package=%s\n' "${expected_package}"
  printf 'package_versions=%s\n' "${package_version}"
  printf 'pyproject_sha256=%s\n' "$(sha256sum -- "${project}/pyproject.toml" | awk '{print $1}')"
  printf 'uv_lock_sha256=%s\n' "$(sha256sum -- "${project}/uv.lock" | awk '{print $1}')"
  printf 'case=%s\n' "${case_path}"
  printf 'case_sha256=%s\n' "$(sha256sum -- "${case_path}" | awk '{print $1}')"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'host=%s\n' "$(hostname)"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)";
} | tee -- "${launch_log}"

args=("${python}" -m fomhd run "${case_path}" --output "${FOMHD_OUTPUT}")
if [[ -n "${FOMHD_RESTART:-}" ]]; then
  args+=(--restart "${FOMHD_RESTART}")
fi
set +e
srun --ntasks=1 "${args[@]}" 2>&1 | tee -- "${FOMHD_OUTPUT}.stdout.log"
statuses=("${PIPESTATUS[@]}")
program_status=${statuses[0]}
tee_status=${statuses[1]}
set -e
status=${program_status}
if [[ "${status}" -eq 0 ]]; then
  status=${tee_status}
fi
set +e
printf 'exit_code=%s\nprogram_exit_code=%s\ntee_exit_code=%s\nfinished_utc=%s\n' \
  "${status}" "${program_status}" "${tee_status}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a -- "${launch_log}"
receipt_statuses=("${PIPESTATUS[@]}")
receipt_status=${receipt_statuses[1]}
set -e
if [[ "${program_status}" -ne 0 ]]; then
  exit "${program_status}"
fi
if [[ "${tee_status}" -ne 0 ]]; then
  exit "${tee_status}"
fi
exit "${receipt_status}"
