#!/usr/bin/env bash
# =============================================================================
# Capacity ablation — SOURCE FD001   (one GPU per source)
#
# Confirms (with proper multi-seed stats) the earlier single-seed grid finding
# that FiLMHeavy/PDMN destabilize as hidden_dim grows: hd32 (main campaign,
# already run) vs hd64 vs hd128, 3 seeds each. Both proposed models have
# ~10-13K params at hd32 vs 300K-1.5M for several baselines (DAGCN, CCDG) --
# this ablation directly answers "is the comparison unfair on capacity?".
#
# LR/dropout/etc. are held at the model's tuned hd32 values (only hidden_dim
# changes, via --hidden_dim_override) so the ablation isolates capacity alone.
#
# 2 models x 2 hidden_dims x 3 transfer pairs x 3 seeds = 36 runs.
# Resume-safe: a run is skipped if its best_checkpoint.pth already exists.
#
# Usage (GPU index as first arg, default 0):
#   bash scripts/run_capacity_ablation_FD001.sh 0
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"

export CUDA_VISIBLE_DEVICES="${1:-0}"

MODELS=("LBGN_RUL" "PDMN")
HIDDEN_DIMS=(64 128)
SRC="FD001"
ALL_TARGETS=("FD002" "FD003" "FD004")
SEEDS=(0 42 64)

total=$(( 3 * ${#MODELS[@]} * ${#HIDDEN_DIMS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  Capacity ablation  |  SOURCE: $SRC  |  GPU $CUDA_VISIBLE_DEVICES  |  $total experiments"
echo "  Models      : ${MODELS[*]}"
echo "  Hidden dims : ${HIDDEN_DIMS[*]}"
echo "  Seeds       : ${SEEDS[*]}"
echo "══════════════════════════════════════════════════════════════"

for tgt in "${ALL_TARGETS[@]}"; do
    [[ "$SRC" == "$tgt" ]] && continue
for model in "${MODELS[@]}"; do
for hd in "${HIDDEN_DIMS[@]}"; do
for seed in "${SEEDS[@]}"; do

    done_count=$(( done_count + 1 ))
    tag="ablation_hd${hd}_s${seed}"
    ckpt="./logs/${SRC}_${tgt}/${model}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $tgt | $model | hd$hd | seed $seed (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $tgt | $model | hd$hd | seed $seed"

    $PY "$SCRIPT" \
        --model_name          "$model"   \
        --dataset_name        CMAPSS     \
        --Data_id_CMAPSS      "$SRC"     \
        --Data_id_CMAPSS_test "$tgt"     \
        --seed                "$seed"    \
        --hidden_dim_override "$hd"      \
        --save_path            "$tag"    \
        --info "${model}_hd${hd}_${SRC}_${tgt}_seed${seed}_ablation" \
        || echo "✗  [FAIL] $SRC → $tgt | $model | hd$hd | seed $seed"

done; done; done; done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Capacity ablation, source $SRC complete.  Results in: logs/"
echo "══════════════════════════════════════════════════════════════"
