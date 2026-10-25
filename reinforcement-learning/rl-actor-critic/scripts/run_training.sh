#!/usr/bin/env bash

# Exit on errors, unset variables, and failed commands inside pipelines.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RL_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"
VENV_DIR="${RL_ROOT}/.venv"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtual environment not found: ${VENV_DIR}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

RUN_LABEL="bash"
FORWARDED_ARGS=("$@")
for ((INDEX = 0; INDEX < ${#FORWARDED_ARGS[@]}; INDEX++)); do
    if [[ "${FORWARDED_ARGS[INDEX]}" == "--run-name" ]] \
        && ((INDEX + 1 < ${#FORWARDED_ARGS[@]})); then
        RUN_LABEL="${FORWARDED_ARGS[INDEX + 1]}"
    fi
done
SAFE_RUN_LABEL="$(printf '%s' "${RUN_LABEL}" | tr -cs 'A-Za-z0-9_.-' '-')"

RUN_STAMP="$(date -u +'%Y%m%dT%H%M%S_%NZ')"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_STAMP}_${SAFE_RUN_LABEL}"
mkdir -p "${RUN_DIR}"

COMMAND=(
    python
    "${PROJECT_ROOT}/scripts/train.py"
    --run-dir "${RUN_DIR}"
    "${FORWARDED_ARGS[@]}"
)

{
    echo "started_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "working_directory=$(pwd)"
    echo "project_root=${PROJECT_ROOT}"
    echo "virtual_environment=${VENV_DIR}"
    python --version
    python -m pip --version
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} | tee "${RUN_DIR}/shell_environment.log"

set +e
"${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/shell_console.log"
TRAINING_STATUS=${PIPESTATUS[0]}
set -e

echo "exit_status=${TRAINING_STATUS}" | tee -a "${RUN_DIR}/shell_environment.log"
echo "finished_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    | tee -a "${RUN_DIR}/shell_environment.log"

exit "${TRAINING_STATUS}"

