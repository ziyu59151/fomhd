#!/usr/bin/env bash
#SBATCH --job-name=fomhd-a0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Submit with explicit paths, for example:
# sbatch --export=ALL,FOMHD_BIN=/abs/build/fomhd,FOMHD_CASE=/abs/case.toml,FOMHD_OUTPUT=/abs/run jobs/a0.slurm
set -euo pipefail

: "${FOMHD_BIN:?Set FOMHD_BIN to the frozen fomhd executable}"
: "${FOMHD_CASE:?Set FOMHD_CASE to the input case.toml}"
: "${FOMHD_OUTPUT:?Set FOMHD_OUTPUT to a new run directory}"

if [[ ! -x "${FOMHD_BIN}" ]]; then
  echo "FOMHD_BIN is not an executable: ${FOMHD_BIN}" >&2
  exit 2
fi
if [[ ! -f "${FOMHD_CASE}" ]]; then
  echo "FOMHD_CASE is not a regular file: ${FOMHD_CASE}" >&2
  exit 2
fi

output_parent=$(dirname -- "${FOMHD_OUTPUT}")
mkdir -p -- "${output_parent}"
launch_log="${FOMHD_OUTPUT}.launch.log"
{
  printf 'binary=%s\n' "${FOMHD_BIN}"
  printf 'binary_sha256=%s\n' "$(sha256sum -- "${FOMHD_BIN}" | awk '{print $1}')"
  printf 'case=%s\n' "${FOMHD_CASE}"
  printf 'case_sha256=%s\n' "$(sha256sum -- "${FOMHD_CASE}" | awk '{print $1}')"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'host=%s\n' "$(hostname)"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -- "${launch_log}"

args=("${FOMHD_BIN}" run "${FOMHD_CASE}" --output "${FOMHD_OUTPUT}")
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
