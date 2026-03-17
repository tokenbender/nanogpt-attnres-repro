#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train.py"
TORCHRUN_BIN="${REPO_ROOT}/.venv/bin/torchrun"

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_BIN="$(command -v torchrun || true)"
fi

# Edit these defaults to match the matrix variant you want to launch.
NPROC_PER_NODE=8
WANDB_PROJECT="nanogpt-attnres-repro"
WANDB_GROUP="fineweb10B-48l-blackwell8-v1"
BATCH_SIZE=8

# Add any shared overrides here, for example:
# EXTRA_OVERRIDES=("compile_model=True")
EXTRA_OVERRIDES=("compile_model=True")

usage() {
    cat <<'EOF'
Usage: run_matrix_48l.sh <algo> [--dry-run]

Algorithms:
  baseline
  mhc
  attnres-full
  attnres-block
  attnres-block-b2
  attnres-block-b8

Examples:
  ./run_matrix_48l.sh baseline
  ./run_matrix_48l.sh attnres-full --dry-run

Edit the variables near the top of this file to change the default W&B project,
group, world size, batch size, or shared overrides.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
fi

ALGO="$1"
DRY_RUN=0

if [[ $# -eq 2 ]]; then
    if [[ "$2" != "--dry-run" ]]; then
        usage
        exit 1
    fi
    DRY_RUN=1
fi

case "${ALGO}" in
    baseline)
        CONFIG="config/train_fineweb10B_48l.py"
        RUN_NAME="baseline-48l"
        OUT_DIR="out-fineweb10B-48l-baseline-blackwell8"
        ALGO_OVERRIDES=()
        ;;
    mhc)
        CONFIG="config/train_fineweb10B_mhc_48l.py"
        RUN_NAME="mhc-48l"
        OUT_DIR="out-fineweb10B-mhc-48l-blackwell8"
        ALGO_OVERRIDES=()
        ;;
    attnres-full)
        CONFIG="config/train_fineweb10B_attnres_full_48l.py"
        RUN_NAME="attnres-full-48l"
        OUT_DIR="out-fineweb10B-attnres-full-48l-blackwell8"
        ALGO_OVERRIDES=()
        ;;
    attnres-block)
        CONFIG="config/train_fineweb10B_attnres_block_48l.py"
        RUN_NAME="attnres-block-b4-48l"
        OUT_DIR="out-fineweb10B-attnres-block-b4-48l-blackwell8"
        ALGO_OVERRIDES=()
        ;;
    attnres-block-b2)
        CONFIG="config/train_fineweb10B_attnres_block_48l.py"
        RUN_NAME="attnres-block-b2-48l"
        OUT_DIR="out-fineweb10B-attnres-block-b2-48l-blackwell8"
        ALGO_OVERRIDES=("attnres_block_size=2")
        ;;
    attnres-block-b8)
        CONFIG="config/train_fineweb10B_attnres_block_48l.py"
        RUN_NAME="attnres-block-b8-48l"
        OUT_DIR="out-fineweb10B-attnres-block-b8-48l-blackwell8"
        ALGO_OVERRIDES=("attnres_block_size=8")
        ;;
    *)
        echo "error: unknown algorithm '${ALGO}'" >&2
        usage
        exit 1
        ;;
esac

if [[ ${DRY_RUN} -eq 0 && ( -z "${TORCHRUN_BIN}" || ! -x "${TORCHRUN_BIN}" ) ]]; then
    echo "error: expected torchrun at ${TORCHRUN_BIN}" >&2
    exit 1
fi

if [[ -z "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_BIN="torchrun"
fi

CMD=(
    "${TORCHRUN_BIN}"
    --standalone
    "--nproc_per_node=${NPROC_PER_NODE}"
    "${TRAIN_PY}"
    "${CONFIG}"
    "batch_size=${BATCH_SIZE}"
    "wandb_project=${WANDB_PROJECT}"
    "wandb_group=${WANDB_GROUP}"
    "wandb_run_name=${RUN_NAME}"
    "out_dir=${OUT_DIR}"
)

if (( ${#EXTRA_OVERRIDES[@]} > 0 )); then
    CMD+=("${EXTRA_OVERRIDES[@]}")
fi

if (( ${#ALGO_OVERRIDES[@]} > 0 )); then
    CMD+=("${ALGO_OVERRIDES[@]}")
fi

echo "Running ${ALGO}"
echo "  config=${CONFIG}"
echo "  wandb_project=${WANDB_PROJECT}"
echo "  wandb_group=${WANDB_GROUP}"
echo "  wandb_run_name=${RUN_NAME}"
echo "  out_dir=${OUT_DIR}"
echo
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ ${DRY_RUN} -eq 1 ]]; then
    exit 0
fi

cd "${SCRIPT_DIR}"
exec "${CMD[@]}"
