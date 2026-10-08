#!/usr/bin/env bash
# =============================================================================
# Final evaluation — SOURCE FD004, all targets, fixed seed list
#
# 9 models x 3 targets x 5 seeds per invocation. Seeds are reapplied here as
# an explicit, fixed list (0 7 42 64 123) -- matching the LBGN_RUL
# final_s0/7/42/64/123 naming already used for the main-table cold-start row
# -- instead of each run drawing an independent random seed, so every model
# in this campaign shares the same seed set for a fair mean+-std comparison.
#
# GPU selection is handled externally (set CUDA_VISIBLE_DEVICES yourself
# before invoking this script) -- not a script argument.
#
# Resume-safe: skips a (target, model, seed) combo if its final_s{seed}
# checkpoint already exists.
#
# Usage:
#   bash scripts/run_final_FD004.sh
#
# Save output:
#   nohup bash scripts/run_final_FD004.sh > final_fd004.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"

SRC="FD004"
ALL_TARGETS=("FD001" "FD002" "FD003")
MODELS=("DAGCN_RUL" "EviAdaptRUL" "DAST_RUL" "CADA_RUL" "TACDA_RUL" "OCS_DANN" "MDAN_RUL" "CCDG_RUL" "LBGN_RUL" "NDC_PDMN" "AIDGN" "DLGNet")
SEEDS=(0 1 7 41 42 64 1234 2026)

total=$(( ${#ALL_TARGETS[@]} * ${#MODELS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  Final runs  |  SOURCE: $SRC  |  $total experiments  |  seeds: ${SEEDS[*]}"
echo "  Targets: ${ALL_TARGETS[*]}"
echo "  Models : ${MODELS[*]}"
echo "══════════════════════════════════════════════════════════════"

for tgt in "${ALL_TARGETS[@]}"; do
    [[ "$SRC" == "$tgt" ]] && continue
for seed in "${SEEDS[@]}"; do
for model in "${MODELS[@]}"; do

    done_count=$(( done_count + 1 ))
    tag="final_s${seed}"
    ckpt="./logs/${SRC}_${tgt}/${model}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $tgt | $model | seed $seed (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $tgt | $model | seed $seed"

    $PY "$SCRIPT" \
        --model_name          "$model"   \
        --dataset_name        CMAPSS     \
        --Data_id_CMAPSS      "$SRC"     \
        --Data_id_CMAPSS_test "$tgt"     \
        --seed                "$seed"    \
        --logs_root            ./logs \
        --save_path            "$tag"    \
        --info                "${model}_${SRC}_${tgt}_seed${seed}_final" \
        || echo "✗  [FAIL] $SRC → $tgt | $model | seed $seed"

done; done; done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Source $SRC complete.  Results in: logs/"
echo "══════════════════════════════════════════════════════════════"
