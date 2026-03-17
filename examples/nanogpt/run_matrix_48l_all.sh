#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_matrix_48l.sh"
LOG_ROOT="${SCRIPT_DIR}/../../experiments/launcher"

# Edit this list if you want a different default sequence.
DEFAULT_ALGOS=(
    attnres-full
    attnres-block
    mhc
    baseline
)

usage() {
    cat <<'EOF'
Usage: run_matrix_48l_all.sh [--dry-run] [algo ...]

Launches the shared 48-layer matrix sequentially using run_matrix_48l.sh.

If no algorithm names are provided, the default sequence is:
  attnres-full attnres-block mhc baseline

Examples:
  ./run_matrix_48l_all.sh
  ./run_matrix_48l_all.sh --dry-run
  ./run_matrix_48l_all.sh baseline mhc

Edit DEFAULT_ALGOS near the top of this file to change the default order.
EOF
}

DRY_RUN=0
ALGOS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ALGOS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -x "${LAUNCHER}" ]]; then
    echo "error: expected launcher at ${LAUNCHER}" >&2
    exit 1
fi

if (( ${#ALGOS[@]} == 0 )); then
    ALGOS=("${DEFAULT_ALGOS[@]}")
fi

mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_ROOT}/run-matrix-48l-${STAMP}.log"

if [[ ${DRY_RUN} -eq 0 ]] && command -v tee >/dev/null 2>&1; then
    exec > >(tee -a "${LOG_PATH}") 2>&1
fi

echo "Sequential matrix launcher"
echo "  dry_run=${DRY_RUN}"
echo "  log=${LOG_PATH}"
echo "  algos=${ALGOS[*]}"
echo

for algo in "${ALGOS[@]}"; do
    echo "[$(date +%F' '%T)] starting ${algo}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        "${LAUNCHER}" "${algo}" --dry-run
    else
        "${LAUNCHER}" "${algo}"
    fi
    echo "[$(date +%F' '%T)] finished ${algo}"
    echo
done

if [[ ${DRY_RUN} -eq 0 ]]; then
    echo "All requested runs completed successfully."
fi
