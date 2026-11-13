#!/usr/bin/env bash
# ============================================================================
# Evaluate a trained MGPD-Net on Landslide4Sense few-shot episodes.
# Colab-friendly; every value can be overridden via environment.
#
# Usage:
#   ./scripts/test_landslide.sh
#   CKPT=.../1000.pth SPLIT=test VAL_STEPS=500 ./scripts/test_landslide.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

# --- Environment detection (Colab vs local) ---------------------------------
IS_COLAB=0
if [[ -d /content ]]; then IS_COLAB=1; fi
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive}"

# --- GPU --------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID:-${GPUID:-0}}"

# --- Paths ------------------------------------------------------------------
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "${DRIVE_ROOT}/thesis/datasets/landslide4sense" ]]; then
    DATA_DIR="${DRIVE_ROOT}/thesis/datasets/landslide4sense"
  elif [[ -d "/content/landslide4sense" ]]; then
    DATA_DIR="/content/landslide4sense"
  else
    DATA_DIR="${REPO_DIR}/datasets/landslide4sense"
  fi
fi
RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  if [[ "${IS_COLAB}" == "1" && -d "${DRIVE_ROOT}" ]]; then
    RUN_DIR="${DRIVE_ROOT}/mgpdnet_runs/landslide_fss_1shot"
  else
    RUN_DIR="${REPO_DIR}/runs/landslide_fss_1shot"
  fi
fi
SPLIT="${SPLIT:-validation}"
CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/checkpoints}"
CKPT="${CKPT:-${RUN_DIR}/final.pth}"
OUT_JSON="${OUT_JSON:-${RUN_DIR}/test_${SPLIT}.json}"

# --- Stage dataset on local disk (avoid heavy Drive random I/O) -------------
# Same rationale as training: evaluation opens h5 patches randomly. On Colab,
# if DATA_DIR is on Drive, copy it once to LOCAL_DATA and evaluate from there.
#   STAGE_DATA=0 -> disable;  FORCE_SYNC=1 -> refresh the local copy.
STAGE_DATA="${STAGE_DATA:-}"
if [[ -z "${STAGE_DATA}" ]]; then
  STAGE_DATA=0
  if [[ "${IS_COLAB}" == "1" && "${DATA_DIR}" == /content/drive/* ]]; then
    STAGE_DATA=1
  fi
fi
LOCAL_DATA="${LOCAL_DATA:-}"
if [[ "${STAGE_DATA}" == "1" && -z "${LOCAL_DATA}" ]]; then
  LOCAL_DATA="/content/landslide4sense"
fi
if [[ -n "${LOCAL_DATA}" ]]; then
  if [[ ! -d "${LOCAL_DATA}" || "${FORCE_SYNC:-0}" == "1" ]]; then
    echo "Staging dataset '${DATA_DIR}' -> '${LOCAL_DATA}' (local disk) ..."
    mkdir -p "${LOCAL_DATA}"
    cp -r "${DATA_DIR%/}/." "${LOCAL_DATA}/"
    echo "  staged $(find "${LOCAL_DATA}" -name '*.h5' | wc -l) .h5 files"
  fi
  DATA_DIR="${LOCAL_DATA%/}"
  echo "Using local dataset: ${DATA_DIR}"
fi

# --- Data / preprocessing (must match training) -----------------------------
NORMALIZE="${NORMALIZE:-imagenet}"
INPUT_SCALE="${INPUT_SCALE:-1.0}"
INPUT_SIZE="${INPUT_SIZE:-${IMG_SIZE:-512}}"

# --- Episode / evaluation (must match training) -----------------------------
N_SHOTS="${N_SHOTS:-${N_SHOT:-1}}"
N_QUERIES="${N_QUERIES:-${N_QUERY:-1}}"
VAL_STEPS="${VAL_STEPS:-${EVAL_EPISODES:-300}}"   # episodes to evaluate
REPEAT="${REPEAT:-1}"                              # resample the eval set N times
SEED="${SEED:-1234}"
NUM_WORKERS="${NUM_WORKERS:-${N_WORKERS:-2}}"
PRETRAINED="${PRETRAINED:-none}"                   # only if ckpt holds no weights
INSTALL_DEPS="${INSTALL_DEPS:-0}"

# --- Optional visualisation -------------------------------------------------
SAVE_VIS="${SAVE_VIS:-}"                           # e.g. /content/vis
VIS_COUNT="${VIS_COUNT:-8}"

# --- Run --------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
mkdir -p "${CKPT_DIR}" "${RUN_DIR}"

VIS_ARGS=()
if [[ -n "${SAVE_VIS}" ]]; then
  mkdir -p "${SAVE_VIS}"
  VIS_ARGS+=(--save-vis "${SAVE_VIS}" --vis-count "${VIS_COUNT}")
fi

echo "========================================================================"
echo " repo        : ${REPO_DIR}"
echo " colab       : ${IS_COLAB}"
echo " data        : ${DATA_DIR}"
echo " ckpt        : ${CKPT}"
echo " json        : ${OUT_JSON}"
echo " split       : ${SPLIT}   shot/query=${N_SHOTS}/${N_QUERIES}   input_size=${INPUT_SIZE}"
echo " episodes    : ${VAL_STEPS} x repeat=${REPEAT}   workers=${NUM_WORKERS}"
echo " gpu         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (gpu_id=${GPU_ID})"
echo "========================================================================"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[error] DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  echo "        set it explicitly, e.g. CKPT=${RUN_DIR}/1000.pth" >&2
  exit 1
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "[info] installing dependencies ..."
  python3 -m pip install -q -r requirements.txt
fi

python3 test.py \
  --data-dir "${DATA_DIR}" \
  --split "${SPLIT}" \
  --checkpoint "${CKPT}" \
  --normalize "${NORMALIZE}" \
  --input-scale "${INPUT_SCALE}" \
  --img-size "${INPUT_SIZE}" \
  --n-shot "${N_SHOTS}" \
  --n-query "${N_QUERIES}" \
  --num-workers "${NUM_WORKERS}" \
  --eval-episodes "${VAL_STEPS}" \
  --repeat "${REPEAT}" \
  --seed "${SEED}" \
  --pretrained "${PRETRAINED}" \
  --ckpt-dir "${CKPT_DIR}" \
  --output-json "${OUT_JSON}" \
  ${VIS_ARGS[@]+"${VIS_ARGS[@]}"} \
  "$@" 2>&1 | tee "${RUN_DIR}/test_${SPLIT}.log"
