#!/bin/bash
# Cluster-specific sbatch script for pi0.5 LIBERO SWAG-LoRA training on the ECE GPU cluster.
#
# Fresh run:
#   sbatch scripts/slurm/sbatch_pi05_libero_swag_lora_h100.sh
#
# Resume an existing experiment:
#   sbatch --export=ALL,EXP_NAME=pi05_libero_swag_lora_seed1,RESUME=1 \
#     scripts/slurm/sbatch_pi05_libero_swag_lora_h100.sh
#
# Useful overrides:
#   sbatch --export=ALL,EXP_NAME=my_run,SEED=3,SWAG_START_STEP=20000,SWAG_COLLECT_INTERVAL=1000,WANDB_MODE=online \
#     scripts/slurm/sbatch_pi05_libero_swag_lora_h100.sh

#SBATCH -J pi05_libero_swag_lora
#SBATCH -p defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_CANDIDATE="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -n "${REPO_ROOT_CANDIDATE}" ]] && git -C "${REPO_ROOT_CANDIDATE}" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "${REPO_ROOT_CANDIDATE}" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi
REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"

CONFIG_NAME="pi05_libero_lora"
DATASET_REPO="physical-intelligence/libero"

EXP_NAME="${EXP_NAME:-pi05_libero_swag_lora_seed1}"
SEED="${SEED:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
WANDB_MODE="${WANDB_MODE:-offline}"
SWAG_START_STEP="${SWAG_START_STEP:-20000}"
SWAG_COLLECT_INTERVAL="${SWAG_COLLECT_INTERVAL:-1000}"
SWAG_MAX_NUM_MODELS="${SWAG_MAX_NUM_MODELS:-10}"
PERSIST_ROOT="${PERSIST_ROOT:-${HOME}/openpi-runs}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PERSIST_ROOT}/checkpoints}"
WANDB_DIR="${WANDB_DIR:-${PERSIST_ROOT}/wandb}"
JOB_SCRATCH="/scratch/${USER}/openpi/${SLURM_JOB_ID}"
JOB_VENV="${JOB_SCRATCH}/.venv"
PERSIST_CACHE_ROOT="${PERSIST_CACHE_ROOT:-${PERSIST_ROOT}/cache}"
PERSIST_HF_HOME="${PERSIST_HF_HOME:-${PERSIST_CACHE_ROOT}/huggingface}"
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p \
    "${JOB_SCRATCH}/tmp" \
    "${JOB_SCRATCH}/openpi-cache" \
    "${JOB_SCRATCH}/xdg-cache" \
    "${JOB_SCRATCH}/matplotlib" \
    "${PERSIST_CACHE_ROOT}/uv" \
    "${PERSIST_HF_HOME}/hub" \
    "${PERSIST_HF_HOME}/lerobot" \
    "${CHECKPOINT_BASE_DIR}" \
    "${WANDB_DIR}"

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi

module purge
module load cuda12.6/toolkit cuda12.6/fft cuda12.6/blas cudnn9.4-cuda12.6

export TMPDIR="${JOB_SCRATCH}/tmp"
export HF_HOME="${PERSIST_HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_LEROBOT_HOME="${HF_HOME}/lerobot"
if [[ -z "${HF_TOKEN:-}" ]]; then
    for TOKEN_FILE in "${HOME}/.cache/huggingface/token" "${HOME}/.huggingface/token"; do
        if [[ -f "${TOKEN_FILE}" ]]; then
            HF_TOKEN="$(tr -d '\r\n' < "${TOKEN_FILE}")"
            export HF_TOKEN
            echo "Loaded HF token from ${TOKEN_FILE}"
            break
        fi
    done
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF token is set for this job"
else
    echo "HF token not found; Hugging Face downloads may be rate-limited"
fi

export OPENPI_DATA_HOME="${JOB_SCRATCH}/openpi-cache"
export UV_CACHE_DIR="${PERSIST_CACHE_ROOT}/uv"
export UV_PROJECT_ENVIRONMENT="${JOB_VENV}"
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="${JOB_SCRATCH}/xdg-cache"
export MPLCONFIGDIR="${JOB_SCRATCH}/matplotlib"
export WANDB_DIR
export WANDB_MODE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export GIT_LFS_SKIP_SMUDGE=1

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs not found; disabling LFS filters for dependency checkout"
    export GIT_CONFIG_COUNT=4
    export GIT_CONFIG_KEY_0=filter.lfs.process
    export GIT_CONFIG_VALUE_0=
    export GIT_CONFIG_KEY_1=filter.lfs.smudge
    export GIT_CONFIG_VALUE_1=
    export GIT_CONFIG_KEY_2=filter.lfs.clean
    export GIT_CONFIG_VALUE_2=cat
    export GIT_CONFIG_KEY_3=filter.lfs.required
    export GIT_CONFIG_VALUE_3=false
fi

cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "Scratch: ${JOB_SCRATCH}"
echo "Checkpoint base: ${CHECKPOINT_BASE_DIR}"
echo "Experiment: ${EXP_NAME}"
echo "Seed: ${SEED}"
echo "W&B mode: ${WANDB_MODE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "SWAG start step: ${SWAG_START_STEP}"
echo "SWAG collect interval: ${SWAG_COLLECT_INTERVAL}"
echo "SWAG max models: ${SWAG_MAX_NUM_MODELS}"
echo "Job venv: ${JOB_VENV}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing to ${HOME}/.local/bin"
    python3 -m pip install --user --upgrade uv
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is still unavailable after installation attempt."
    exit 1
fi

echo "uv: $(command -v uv)"
uv --version

echo "Creating job-local environment from pyproject/uv.lock"
uv python install 3.11
uv sync --frozen --no-dev

PYTHON_BIN="${JOB_VENV}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Expected job python at ${PYTHON_BIN} after uv sync, but it was not created."
    exit 1
fi

echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

nvidia-smi
"${PYTHON_BIN}" - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY

NORM_STATS_PATH="${REPO_ROOT}/assets/${CONFIG_NAME}/${DATASET_REPO}/norm_stats.json"
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
    echo "Normalization stats not found at ${NORM_STATS_PATH}"
    echo "Computing norm stats for ${CONFIG_NAME}"
    "${PYTHON_BIN}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
else
    echo "Using existing norm stats: ${NORM_STATS_PATH}"
fi

RUN_MODE_FLAG="--overwrite"
if [[ "${RESUME:-0}" == "1" ]]; then
    RUN_MODE_FLAG="--resume"
fi

echo "Starting SWAG-LoRA training"
"${PYTHON_BIN}" scripts/train_pi05_libero_swag_lora.py \
    --exp-name "${EXP_NAME}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --swag-start-step "${SWAG_START_STEP}" \
    --swag-collect-interval "${SWAG_COLLECT_INTERVAL}" \
    --swag-max-num-models "${SWAG_MAX_NUM_MODELS}" \
    "${RUN_MODE_FLAG}"

echo "Training finished"
echo "Checkpoints: ${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
echo "SWAG state: ${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/swag_lora_state.pkl"
