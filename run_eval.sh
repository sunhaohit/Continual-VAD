#!/usr/bin/env bash
set -euo pipefail

# Final fixed-parameter evaluation entry.
# Usage:
#   GPU=0 TASK=sh_full bash run_eval.sh
#   GPU=1 TASK=sh_inc bash run_eval.sh
#   GPU=2 TASK=ucf_full bash run_eval.sh
#   GPU=3 TASK=ucf_inc bash run_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-${SCRIPT_DIR}}"
WORKDIR="${WORKDIR:-${BASE}}"
TASK="${TASK:-${1:-}}"
GPU="${GPU:-0}"

if [[ -z "${TASK}" ]]; then
  echo "[ERR] TASK is required: sh_full | sh_inc | ucf_full | ucf_inc" >&2
  exit 2
fi

cd "${WORKDIR}"
for f in eval.py models.py options.py video_dataset_anomaly_balance_uni_sample_ucf.py utils.py; do
  [[ -f "${f}" ]] || { echo "[ERR] missing ${WORKDIR}/${f}" >&2; exit 1; }
done

SH_ROOT="${SH_ROOT:-}"
SH_DIL_ROOT="${SH_DIL_ROOT:-}"
SH_FULL_TRAIN="${SH_FULL_TRAIN:-}"
SH_INC_TRAIN="${SH_INC_TRAIN:-}"
SH_FULL_CKPT="${SH_FULL_CKPT:-}"
SH_INC_CKPT="${SH_INC_CKPT:-}"

UCF_ROOT="${UCF_ROOT:-}"
UCF_DIL_ROOT="${UCF_DIL_ROOT:-}"
UCF_FULL_TRAIN="${UCF_FULL_TRAIN:-}"
UCF_INC_TRAIN="${UCF_INC_TRAIN:-}"
UCF_FULL_CKPT="${UCF_FULL_CKPT:-}"
UCF_INC_CKPT="${UCF_INC_CKPT:-}"

require_path() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "[ERR] ${name} is required for TASK=${TASK}. Set it as an environment variable." >&2
    exit 2
  fi
}

case "${TASK}" in
  sh_full)
    require_path "SH_ROOT" "${SH_ROOT}"
    require_path "SH_DIL_ROOT" "${SH_DIL_ROOT}"
    require_path "SH_FULL_TRAIN" "${SH_FULL_TRAIN}"
    DATASET_NAME="shanghaitech"
    DATASET_KIND="sh"
    DATASET_PATH="${SH_ROOT}"
    DIL_ROOT="${dil_root:-${SH_DIL_ROOT}}"
    TRAIN_NPY="${train_npy:-${SH_FULL_TRAIN}}"
    INPUT_NORM="${input_norm:-none}"
    INC_ARGS=(--inc_enable False)
    HIDDEN1="${scorer_hidden1:-1024}"
    HIDDEN2="${scorer_hidden2:-128}"
    DEFAULT_CKPT="${SH_FULL_CKPT:-${BASE}/bestckpt/sh_full.pkl}"
    SMOOTH=13
    MIX=0.0
    BASE_Q=55
    STRENGTH=0.45
    EVAL_MODE="standard"
    ;;
  sh_inc)
    require_path "SH_ROOT" "${SH_ROOT}"
    require_path "SH_DIL_ROOT" "${SH_DIL_ROOT}"
    DATASET_NAME="shanghaitech"
    DATASET_KIND="sh"
    DATASET_PATH="${SH_ROOT}"
    DIL_ROOT="${dil_root:-${SH_DIL_ROOT}}"
    TRAIN_NPY="${train_npy:-${SH_INC_TRAIN:-${SH_DIL_ROOT}/scene01/train_unlabel.npy}}"
    INPUT_NORM="${input_norm:-l2}"
    INC_ARGS=(--inc_enable True --inc_scene_id "${inc_scene_id:-12}" --inc_expected_scenes "${inc_expected_scenes:-13}" --inc_memory_max "${inc_memory_max:-512}" --inc_scene_proto_cap "${inc_scene_proto_cap:-64}" --inc_scene_min_slots "${inc_scene_min_slots:-16}")
    HIDDEN1="${scorer_hidden1:-1024}"
    HIDDEN2="${scorer_hidden2:-128}"
    DEFAULT_CKPT="${SH_INC_CKPT:-${BASE}/bestckpt/sh_incremental.pkl}"
    SMOOTH=21
    MIX=0.25
    BASE_Q=75
    STRENGTH=0.70
    EVAL_MODE="standard"
    ;;
  ucf_full)
    require_path "UCF_ROOT" "${UCF_ROOT}"
    require_path "UCF_DIL_ROOT" "${UCF_DIL_ROOT}"
    require_path "UCF_FULL_TRAIN" "${UCF_FULL_TRAIN}"
    DATASET_NAME="ucfcrime"
    DATASET_KIND="ucf"
    DATASET_PATH="${UCF_ROOT}"
    DIL_ROOT="${dil_root:-${UCF_DIL_ROOT}}"
    TRAIN_NPY="${train_npy:-${UCF_FULL_TRAIN}}"
    INPUT_NORM="${input_norm:-none}"
    INC_ARGS=(--inc_enable False)
    HIDDEN1="${scorer_hidden1:-512}"
    HIDDEN2="${scorer_hidden2:-32}"
    DEFAULT_CKPT="${UCF_FULL_CKPT:-${BASE}/bestckpt/ucf_full.pkl}"
    SMOOTH=13
    MIX=0.0
    BASE_Q=55
    STRENGTH=0.45
    EVAL_MODE="standard"
    ;;
  ucf_inc)
    require_path "UCF_ROOT" "${UCF_ROOT}"
    require_path "UCF_DIL_ROOT" "${UCF_DIL_ROOT}"
    DATASET_NAME="ucfcrime"
    DATASET_KIND="ucf"
    DATASET_PATH="${UCF_ROOT}"
    DIL_ROOT="${dil_root:-${UCF_DIL_ROOT}}"
    TRAIN_NPY="${train_npy:-${UCF_INC_TRAIN:-${UCF_DIL_ROOT}/domain01/train_unlabel.npy}}"
    INPUT_NORM="${input_norm:-none}"
    INC_ARGS=(--inc_enable True --inc_scene_id "${inc_scene_id:-8}" --inc_expected_scenes "${inc_expected_scenes:-9}" --inc_memory_max "${inc_memory_max:-320}" --inc_scene_proto_cap "${inc_scene_proto_cap:-40}" --inc_scene_min_slots "${inc_scene_min_slots:-10}")
    HIDDEN1="${scorer_hidden1:-1024}"
    HIDDEN2="${scorer_hidden2:-128}"
    DEFAULT_CKPT="${UCF_INC_CKPT:-${BASE}/bestckpt/ucf_incremental.pkl}"
    SMOOTH=7
    MIX=0.12
    BASE_Q=45
    STRENGTH=0.70
    EVAL_MODE="ucf_inc_fusion"
    ;;
  *)
    echo "[ERR] unknown TASK=${TASK}" >&2
    exit 2
    ;;
esac

CKPT="${CKPT:-${ckpt:-}}"
CKPT="${CKPT:-${DEFAULT_CKPT}}"
[[ -f "${CKPT}" ]] || { echo "[ERR] checkpoint file is missing for TASK=${TASK}" >&2; exit 1; }
[[ -d "${DIL_ROOT}" ]] || { echo "[ERR] DIL_ROOT is missing or not a directory for TASK=${TASK}" >&2; exit 1; }
[[ -e "${TRAIN_NPY}" ]] || { echo "[ERR] train feature npy is missing for TASK=${TASK}" >&2; exit 1; }
[[ -e "${DIL_ROOT}/global_test.npy" ]] || { echo "[ERR] global_test.npy is missing under DIL_ROOT for TASK=${TASK}" >&2; exit 1; }
[[ -e "${DIL_ROOT}/global_test_gt.npy" ]] || { echo "[ERR] global_test_gt.npy is missing under DIL_ROOT for TASK=${TASK}" >&2; exit 1; }
[[ -e "${DIL_ROOT}/global_build_meta.json" ]] || { echo "[ERR] global_build_meta.json is missing under DIL_ROOT for TASK=${TASK}" >&2; exit 1; }

OUT_DIR="${OUT_DIR:-${BASE}/results/${TASK}}"
mkdir -p "${OUT_DIR}"

python eval.py \
  --task "${TASK}" \
  --eval_mode "${EVAL_MODE}" \
  --device "${GPU}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_path "${DATASET_PATH}" \
  --data_mode concat \
  --ucf_train_npy "${TRAIN_NPY}" \
  --feature_size 512 \
  --proj_feature_size 0 \
  --input_norm "${INPUT_NORM}" \
  --Vitblock_num 8 \
  --cross_clip 2 \
  --scorer_hidden1 "${HIDDEN1}" \
  --scorer_hidden2 "${HIDDEN2}" \
  --scorer_dropout 0.7 \
  --concat_test_use_cross False \
  --concat_test_score_mode first \
  --use_qavclab False \
  "${INC_ARGS[@]}" \
  --ckpt_path "${CKPT}" \
  --dil_root "${DIL_ROOT}" \
  --dil_dataset "${DATASET_KIND}" \
  --infer_with_teacher False \
  --infer_with_anchor False \
  --infer_with_memory_novelty False \
  --infer_with_recon False \
  --use_eval_postproc True \
  --test_smooth_kernel "${SMOOTH}" \
  --test_video_baseline_mix "${MIX}" \
  --test_video_baseline_q "${BASE_Q}" \
  --test_video_baseline_strength "${STRENGTH}" \
  --test_video_baseline_rescale True \
  --batch_eval 128 \
  --output_json "${OUT_DIR}/eval.json" \
  --output_csv "${OUT_DIR}/eval.csv"

echo "[DONE] eval task=${TASK} ckpt=$(basename "${CKPT}") out=results/${TASK}"
