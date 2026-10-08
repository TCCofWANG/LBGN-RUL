#!/usr/bin/env bash
# =============================================================================
# NDC_PDMN r≡1 ablation: same architecture, same hparams, same loss terms
# (calibration + EOL) as the full model -- the ONLY difference is that the
# damage-clock rate is frozen at the constant 1 (tau collapses to raw OCC
# exactly), via --ndc_ablate_rate1. This isolates the effect of the LEARNED
# clock from everything else in the training recipe.
#
# Covers all 12 source->target pairs, 5 seeds each (matching the main
# NDC_PDMN campaign seeds), so the two result sets form a clean paired
# comparison (Wilcoxon over n=60).
#
# Deliberately does NOT tag --info with "_final" -- these rows must never be
# picked up by final_table.py / add_pdmn_ndc_to_xlsx.py's "_final" filter,
# so the main results table can never be silently contaminated by ablation
# runs. Checkpoints go to a separate ablation_r1_s{seed} dir per pair, so
# they can't collide with or overwrite the real final_s{seed} checkpoints.
#
# Resume-safe: a run is skipped if its best_checkpoint.pth already exists.
#
# Usage (GPU index as first arg, default 0):
#   bash scripts/run_ablation_ndc_r1.sh 0
#
# Save output:
#   nohup bash scripts/run_ablation_ndc_r1.sh 0 > ablation_ndc_r1.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"

MODEL="NDC_PDMN"
SEEDS=(0 42 64 7 123)

PAIRS=(
    "FD001 FD002" "FD001 FD003" "FD001 FD004"
    "FD002 FD001" "FD002 FD003" "FD002 FD004"
    "FD003 FD001" "FD003 FD002" "FD003 FD004"
    "FD004 FD001" "FD004 FD002" "FD004 FD003"
)

total=$(( ${#PAIRS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  NDC_PDMN r≡1 ablation  |  GPU $CUDA_VISIBLE_DEVICES  |  $total experiments"
echo "  Seeds  : ${SEEDS[*]}"
echo "══════════════════════════════════════════════════════════════"

for pair in "${PAIRS[@]}"; do
    read -r SRC tgt <<< "$pair"
for seed in "${SEEDS[@]}"; do

    done_count=$(( done_count + 1 ))
    tag="ablation_r1_s${seed}"
    ckpt="./logs/${SRC}_${tgt}/${MODEL}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $tgt | seed $seed (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $tgt | seed $seed"

    $PY "$SCRIPT" \
        --model_name          "$MODEL"  \
        --dataset_name        CMAPSS    \
        --Data_id_CMAPSS      "$SRC"    \
        --Data_id_CMAPSS_test "$tgt"    \
        --seed                "$seed"   \
        --save_path            "$tag"   \
        --ndc_ablate_rate1               \
        --info                 "${MODEL}_ablr1_${SRC}_${tgt}_seed${seed}_ablation" \
        || echo "✗  [FAIL] $SRC → $tgt | seed $seed"

done; done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  r≡1 ablation complete.  Results in: logs/  (info tag: _ablation, NOT _final)"
echo "════════════════════════════════════════════════════════════════"
