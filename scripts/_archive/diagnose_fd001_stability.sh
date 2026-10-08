#!/usr/bin/env bash
# =============================================================================
# Diagnostic: is FiLMHeavy's FD001-source instability (score blow-ups on
# FD001->FD003 esp.) a training-protocol artifact or a real architectural
# weakness?
#
# Runs LBGN_RUL on all 3 FD001-source pairs x 5 seeds under two
# configs:
#   CURRENT      : lr=5e-3 (config default), early_stop_patience=15 (default)
#   CONSERVATIVE : lr=1e-3, early_stop_patience=25
#
# 3 pairs x 5 seeds x 2 configs = 30 short runs (~13K-param model, fast).
# Tagged separately from the main final_s* campaign -- does NOT touch or
# overwrite those results.
#
# Usage:
#   bash scripts/diagnose_fd001_stability.sh 0   # GPU index
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${1:-0}"
PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"

MODEL="LBGN_RUL"
SRC="FD001"
TARGETS=("FD002" "FD003" "FD004")
SEEDS=(0 42 64 7 123)

total=$(( 2 * ${#TARGETS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  Stability diagnostic  |  SOURCE: $SRC  |  MODEL: $MODEL  |  $total runs"
echo "  current:      lr=5e-3 (config default), patience=15 (default)"
echo "  conservative:  lr=1e-3,                  patience=25"
echo "══════════════════════════════════════════════════════════════"

for tgt in "${TARGETS[@]}"; do
for seed in "${SEEDS[@]}"; do
for variant in current conservative; do

    done_count=$(( done_count + 1 ))
    tag="diag_${variant}_s${seed}"
    ckpt="./logs/${SRC}_${tgt}/${MODEL}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $tgt | $variant | seed $seed (done)"
        continue
    fi

    if [[ "$variant" == "conservative" ]]; then
        LR_ARGS="--learning_rate_override 1e-3 --early_stop_patience 25"
    else
        LR_ARGS=""
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $tgt | $variant | seed $seed"

    $PY "$SCRIPT" \
        --model_name          "$MODEL"  \
        --dataset_name        CMAPSS    \
        --Data_id_CMAPSS      "$SRC"    \
        --Data_id_CMAPSS_test "$tgt"    \
        --seed                "$seed"   \
        --save_path            "$tag"   \
        --info                "${MODEL}_${SRC}_${tgt}_${variant}_seed${seed}_diag" \
        $LR_ARGS \
        || echo "✗  [FAIL] $SRC → $tgt | $variant | seed $seed"

done; done; done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Diagnostic complete.  Compare with:"
echo "  python analysis/diagnose_stability_report.py"
echo "══════════════════════════════════════════════════════════════"
