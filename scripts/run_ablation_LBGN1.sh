#!/usr/bin/env bash
# =============================================================================
# LBGN ablation study — SOURCE FD001, all targets, fixed seed list
#
# 15 variants x 3 targets x 8 seeds per invocation. Same fixed seed list as
# scripts/run_final_FD001.sh (0 1 7 41 42 64 1234 2026), so the ablation's
# 'none' reference can eventually be compared seed-for-seed against the main
# table's LBGN_RUL row. Protocol: COLD START for every variant INCLUDING
# 'none' -- ablated architectures can't share a warm-start checkpoint, so the
# reference row is never warm-started either (keeps the ablation internally
# consistent, even though the main table's LBGN row is warm-started -- see
# CMAPSS_DA.py's --resume_path default).
#
#   RQ1  spectral substrate      : time_adj      vs none
#   RQ2  lifecycle band selection: uniform_band / static_band / no_lifecycle vs none
#   RQ3  FiLM at all             : no_film       vs none
#   RQ4  depth-wise independence : shared_film / no_input_film vs none
#   RQ5  spectral-view necessity : no_spectral_view / no_spectral_film /
#                                 no_spectral_diffusion vs none
#   loss : plain_mse / no_mar / no_mmd / no_da  vs none
#
# Hyperparameters come from configs/hparams.py (identical to final runs);
# 'ablation' in --info keeps these rows OUT of the main results table.
#
# GPU selection is handled externally (set CUDA_VISIBLE_DEVICES yourself
# before invoking this script) -- not a script argument.
#
# Resume-safe: skips a (target, variant, seed) combo if its abl_{variant}_s{seed}
# checkpoint already exists.
#
# Usage:
#   bash scripts/run_ablation_LBGN1.sh
#
# Save output:
#   nohup bash scripts/run_ablation_LBGN1.sh > ablation_fd001.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
MODEL="LBGN_RUL"

SRC="FD001"
ALL_TARGETS=("FD002" "FD003" "FD004")
VARIANTS=("none" "time_adj" "uniform_band" "static_band" "no_lifecycle" "no_input_film" "shared_film" "no_film" "no_spectral_view" "no_spectral_film" "no_spectral_diffusion" "uniform_band_no_spectral_diffusion" "plain_mse" "no_mar" "no_mmd" "no_da")
SEEDS=(0 1 7 41 42 64 1234 2026)

total=$(( ${#ALL_TARGETS[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  LBGN ablation  |  SOURCE: $SRC  |  $total experiments  |  seeds: ${SEEDS[*]}"
echo "  Targets : ${ALL_TARGETS[*]}"
echo "  Variants: ${VARIANTS[*]}"
echo "══════════════════════════════════════════════════════════════"

for tgt in "${ALL_TARGETS[@]}"; do
    [[ "$SRC" == "$tgt" ]] && continue
for seed in "${SEEDS[@]}"; do
for variant in "${VARIANTS[@]}"; do

    done_count=$(( done_count + 1 ))
    tag="abl_${variant}_s${seed}"
    ckpt="./abl_logs/${SRC}_${tgt}/${MODEL}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $tgt | $variant | seed $seed (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $tgt | $variant | seed $seed"

    $PY "$SCRIPT" \
        --model_name          "$MODEL"   \
        --dataset_name        CMAPSS     \
        --Data_id_CMAPSS      "$SRC"     \
        --Data_id_CMAPSS_test "$tgt"     \
        --seed                "$seed"    \
        --logs_root           ./abl_logs \
        --save_path            "$tag"    \
        --lbgn_ablation        "$variant" \
        --info                "${MODEL}_${SRC}_${tgt}_seed${seed}_ablation_${variant}" \
        || echo "✗  [FAIL] $SRC → $tgt | $variant | seed $seed"

done; done; done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Source $SRC ablation complete.  Results in: abl_logs/"
echo "══════════════════════════════════════════════════════════════"
