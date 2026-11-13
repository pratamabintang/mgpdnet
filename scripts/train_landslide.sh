#!/usr/bin/env bash
# ============================================================================
# Train MGPD-Net on the Landslide4Sense dataset (RGB B2/B3/B4 few-shot, binary
# landslide). Colab-friendly; every value can be overridden via environment.
#
# Usage:
#   ./scripts/train_landslide.sh              # defaults below
#   N_STEPS=5000 ./scripts/train_landslide.sh # override via env
#
# Env var names mirror the PANet train.sh where they mean the same thing.
# NOTE: MGPD-Net (FewShotSeg) only supports BATCH_SIZE=1.
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

# --- Run / task settings ----------------------------------------------------
SEED="${SEED:-1234}"
N_STEPS="${N_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"            # must stay 1 for MGPD-Net
NUM_WORKERS="${NUM_WORKERS:-${N_WORKERS:-8}}"
N_WAYS="${N_WAYS:-1}"
N_SHOTS="${N_SHOTS:-${N_SHOT:-1}}"
N_QUERIES="${N_QUERIES:-${N_QUERY:-1}}"
LR="${LR:-1e-3}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-${WD:-5e-4}}"
LR_GAMMA="${LR_GAMMA:-0.80}"
AMP="${AMP:-auto}"                       # auto|fp16|bf16|none
VAL_INTERVAL="${VAL_INTERVAL:-${EVAL_INTERVAL:-1000}}"  # steps between val (0=off)
VAL_STEPS="${VAL_STEPS:-${EVAL_EPISODES:-100}}"         # val episodes per run
INPUT_SIZE="${INPUT_SIZE:-${IMG_SIZE:-128}}"            # square resize
MAX_ITER="${MAX_ITER:-1000}"             # epoch size (episodes per reload)

# --- Loss weights -----------------------------------------------------------
ALIGN_W="${ALIGN_W:-1.0}"
MSE_W="${MSE_W:-1.0}"
CONTRA_W="${CONTRA_W:-1.0}"

# --- Augmentation (landslide spectral only) ---------------------------------
FLIP="${FLIP:-${AUG_FLIP:-True}}"        # horizontal/vertical random flip
ROT90="${ROT90:-${AUG_ROT90:-True}}"     # k*90 rotations
ROTATION="${ROTATION:-0}"                # max |angle| degrees, 0 disables
COLOR="${COLOR:-${AUG_COLOR:-True}}"     # photometric jitter (spectral bands)
COLOR_BRIGHTNESS="${COLOR_BRIGHTNESS:-0.2}"
COLOR_CONTRAST="${COLOR_CONTRAST:-0.2}"
COLOR_GAMMA="${COLOR_GAMMA:-0.2}"
COLOR_GAIN="${COLOR_GAIN:-0.1}"

# --- Data / checkpoint ------------------------------------------------------
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "${DRIVE_ROOT}/thesis/datasets/landslide4sense" ]]; then
    DATA_DIR="${DRIVE_ROOT}/thesis/datasets/landslide4sense"
  elif [[ -d "/content/landslide4sense" ]]; then
    DATA_DIR="/content/landslide4sense"
  else
    DATA_DIR="${REPO_DIR}/datasets/landslide4sense"
  fi
fi
SPLIT="${SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-validation}"
NORMALIZE="${NORMALIZE:-imagenet}"
INPUT_SCALE="${INPUT_SCALE:-1.0}"        # scale raw bands before normalization
CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/checkpoints}"
OUT_DIR="${OUT_DIR:-}"
if [[ -z "${OUT_DIR}" ]]; then
  if [[ "${IS_COLAB}" == "1" && -d "${DRIVE_ROOT}" ]]; then
    OUT_DIR="${DRIVE_ROOT}/mgpdnet_runs/landslide_fss_1shot"
  else
    OUT_DIR="${REPO_DIR}/runs/landslide_fss_1shot"
  fi
fi
LOG_DIR="${LOG_DIR:-$(dirname "${OUT_DIR}")}"

# --- Pretrained encoder weights ---------------------------------------------
# PRETRAINED: 'deeplabv3' | 'resnet101' | 'none' | path to a .pth.
# INIT_PATH (optional, PANet-style): an explicit local checkpoint path.
PRETRAINED="${PRETRAINED:-deeplabv3}"
INIT_PATH="${INIT_PATH:-}"
if [[ -n "${INIT_PATH}" && -f "${INIT_PATH}" ]]; then
  PRETRAINED="${INIT_PATH}"
fi

# --- Bookkeeping ------------------------------------------------------------
PRINT_INTERVAL="${PRINT_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${SNAPSHOT_INTERVAL:-1000}}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"        # 1 -> pip install -r requirements.txt

# --- Stage dataset on local disk (avoid heavy Drive random I/O) -------------
# Training reads h5 patches randomly (image+mask per episode, plus a foreground
# probe per sample); doing that against Google Drive is slow. On Colab, if
# DATA_DIR lives on Drive, it is copied once to LOCAL_DATA (default
# /content/landslide4sense) and training runs from there.
#   STAGE_DATA=0   -> disable (train straight from DATA_DIR)
#   FORCE_SYNC=1   -> refresh the local copy
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

# --- Assemble ---------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
mkdir -p "${OUT_DIR}" "${CKPT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log}"

AUG_ARGS=()
if [[ "${FLIP}" == "True"  || "${FLIP}" == "1"  ]]; then AUG_ARGS+=(--aug-flip);  fi
if [[ "${ROT90}" == "True" || "${ROT90}" == "1" ]]; then AUG_ARGS+=(--aug-rot90); fi
if [[ "${COLOR}" == "True" || "${COLOR}" == "1" ]]; then AUG_ARGS+=(--aug-color); fi

echo "========================================================================"
echo " repo        : ${REPO_DIR}"
echo " colab       : ${IS_COLAB}"
echo " data        : ${DATA_DIR}"
echo " out         : ${OUT_DIR}"
echo " ckpt        : ${CKPT_DIR}"
echo " log         : ${LOG_FILE}"
echo " gpu         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (gpu_id=${GPU_ID})"
echo " shot/query  : ${N_SHOTS}/${N_QUERIES}   ways=${N_WAYS}   input_size=${INPUT_SIZE}"
echo " steps       : ${N_STEPS}  (epoch=${MAX_ITER})  workers=${NUM_WORKERS}"
echo " optim       : lr=${LR} wd=${WEIGHT_DECAY} gamma=${LR_GAMMA} amp=${AMP}"
echo " aug         : flip=${FLIP} rot90=${ROT90} rotation=${ROTATION} color=${COLOR}"
echo " pretrained  : ${PRETRAINED}"
echo "========================================================================"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[error] DATA_DIR does not exist: ${DATA_DIR}" >&2
  echo "        set it explicitly, e.g. DATA_DIR=/content/drive/MyDrive/thesis/datasets/landslide4sense" >&2
  exit 1
fi
if [[ "${BATCH_SIZE}" != "1" ]]; then
  echo "[warn] MGPD-Net only supports BATCH_SIZE=1; got ${BATCH_SIZE}. Forcing 1." >&2
  BATCH_SIZE=1
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "[info] installing dependencies ..."
  python3 -m pip install -q -r requirements.txt
fi

python3 train.py \
  --data-dir "${DATA_DIR}" \
  --split "${SPLIT}" \
  --val-split "${VAL_SPLIT}" \
  --normalize "${NORMALIZE}" \
  --input-scale "${INPUT_SCALE}" \
  --img-size "${INPUT_SIZE}" \
  --n-shot "${N_SHOTS}" \
  --n-query "${N_QUERIES}" \
  --num-workers "${NUM_WORKERS}" \
  --n-steps "${N_STEPS}" \
  --max-iters-per-load "${MAX_ITER}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --momentum "${MOMENTUM}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lr-step-gamma "${LR_GAMMA}" \
  --align-weight "${ALIGN_W}" \
  --mse-weight "${MSE_W}" \
  --contrastive-weight "${CONTRA_W}" \
  --amp "${AMP}" \
  --aug-rotation "${ROTATION}" \
  --color-brightness "${COLOR_BRIGHTNESS}" \
  --color-contrast "${COLOR_CONTRAST}" \
  --color-gamma "${COLOR_GAMMA}" \
  --color-gain "${COLOR_GAIN}" \
  --pretrained "${PRETRAINED}" \
  --ckpt-dir "${CKPT_DIR}" \
  --seed "${SEED}" \
  --print-interval "${PRINT_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --eval-interval "${VAL_INTERVAL}" \
  --eval-episodes "${VAL_STEPS}" \
  --output-dir "${OUT_DIR}" \
  ${AUG_ARGS[@]+"${AUG_ARGS[@]}"} \
  "$@" 2>&1 | tee "${LOG_FILE}"
