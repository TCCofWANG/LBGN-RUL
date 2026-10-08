#!/usr/bin/env bash
# =============================================================================
# Targeted LR fix for two baselines on FD003 -> FD001 specifically.
#
# The source-level majority-vote LR (one LR per source, applied to all its
# targets) is a poor fit for these two (model, pair) combos: ALL surviving
# seeds were consistently bad, not just isolated outliers -- the single-seed
# grid's "good" number at the current LR was a fluke. A different LR (still
# from the same original grid, just not the majority pick) is clearly
# better for this specific pair:
#   DAGCN_RUL : 0.0005 (source default) -> 0.001  (grid single-seed: 18.87-implied-fluke vs 22.80 more plausible)
#   CCDG_RUL  : 0.001  (source default) -> 0.005  (grid single-seed: 30.95 vs 26.58)
#
# FD003's OTHER targets (FD003->FD002, FD003->FD004) are NOT touched --
# they are already well served by the source-level default LR.
#
# All 5 standard seeds are rerun here (not just the flagged anomaly), since
# the whole combo was systematically off, not just 1-2 unlucky seeds.
#
# Usage: bash scripts/rerun_lr_fix_FD003_FD001.sh 0   # GPU index
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
export CUDA_VISIBLE_DEVICES="${1:-0}"

SEEDS=(0 42 64 7 123)

for seed in "${SEEDS[@]}"; do
    echo "▶ DAGCN_RUL FD003->FD001 seed$seed (lr=0.001 override)"
    $PY CMAPSS_DA.py --model_name DAGCN_RUL --dataset_name CMAPSS \
        --Data_id_CMAPSS FD003 --Data_id_CMAPSS_test FD001 \
        --seed "$seed" --learning_rate_override 0.001 \
        --save_path "final_s${seed}" \
        --info "DAGCN_RUL_FD003_FD001_seed${seed}_final_lrfix" \
        || echo "✗ FAIL DAGCN_RUL seed$seed"
done

for seed in "${SEEDS[@]}"; do
    echo "▶ CCDG_RUL FD003->FD001 seed$seed (lr=0.005 override)"
    $PY CMAPSS_DA.py --model_name CCDG_RUL --dataset_name CMAPSS \
        --Data_id_CMAPSS FD003 --Data_id_CMAPSS_test FD001 \
        --seed "$seed" --learning_rate_override 0.005 \
        --save_path "final_s${seed}" \
        --info "CCDG_RUL_FD003_FD001_seed${seed}_final_lrfix" \
        || echo "✗ FAIL CCDG_RUL seed$seed"
done

echo "Done. These overwrite final_s{seed} for DAGCN_RUL/CCDG_RUL on FD003_FD001 only."
