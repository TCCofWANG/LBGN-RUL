#!/usr/bin/env bash
# =============================================================================
# Cross-domain DA on N-CMAPSS -- one source, ALL models x ALL targets x a
# fixed seed list.
#
# Seeds are a hardcoded, fixed list (0 1 7 41 42 64 1234 2026) matching the
# CMAPSS run_final_FD00{1,2,3,4}.sh convention, instead of taking one seed
# per invocation -- so every model/pair here shares the same seed set for a
# fair mean+-std comparison, and scripts/copy_best_checkpoints.py has
# multiple seeds per (pair, model) to actually pick a "best" one from.
#
# N-CMAPSS support in this codebase was single-dataset only before this
# change -- Experiment.py had no source->target branch for it. This script
# exercises the new --Data_id_N_CMAPSS_test / _get_data_da_ncmapss() path.
#
# configs/hparams.py's N_CMAPSS entries for every model except DLGNet are
# UNTESTED (copied unchanged from CMAPSS's FD001 block, num_nodes 14->20) --
# treat any result from this script as a first look, not tuned numbers.
#
# Resume-safe: skips a (model, target, seed) combo if its best_checkpoint.pth
# already exists.
#
# Usage:
#   bash scripts/NCMAPSS/run_ncmapss_da.sh <SRC>
#   bash scripts/NCMAPSS/run_ncmapss_da.sh DS01
#    SEEDS=(0 1 7 41 42 64 1234 2026)
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SRC="${1:?usage: run_ncmapss_da.sh SRC}"

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"

MODELS=("LBGN_RUL" "NDC_PDMN" "DAGCN_RUL" "EviAdaptRUL" "DAST_RUL" "CADA_RUL" "TACDA_RUL" "OCS_DANN" "MDAN_RUL" "CCDG_RUL" "AIDGN" "DLGNet")
SEEDS=(0 1 7 41 42 64 1234 2026)

ALL_DS=("DS01" "DS02" "DS03" "DS04")
TARGETS=()
for ds in "${ALL_DS[@]}"; do
    [[ "$ds" == "$SRC" ]] && continue
    TARGETS+=("$ds")
done

total=$(( ${#MODELS[@]} * ${#TARGETS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  N-CMAPSS DA  |  SRC: $SRC  |  $total experiments  |  seeds: ${SEEDS[*]}"
echo "  Models : ${MODELS[*]}"
echo "  Targets: ${TARGETS[*]}"
echo "══════════════════════════════════════════════════════════════"

for model in "${MODELS[@]}"; do
for TGT   in "${TARGETS[@]}"; do
for SEED  in "${SEEDS[@]}"; do

    done_count=$(( done_count + 1 ))
    tag="final_s${SEED}"
    ckpt="./logs_NCMAPSS/${SRC}_${TGT}/${model}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $TGT | $model | seed $SEED (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $TGT | $model | seed $SEED"

    $PY "$SCRIPT" \
        --model_name             "$model"  \
        --dataset_name           N_CMAPSS  \
        --Data_id_N_CMAPSS       "$SRC"    \
        --Data_id_N_CMAPSS_test  "$TGT"    \
        --seed                   "$SEED"   \
        --train                   True     \
        --resume                  True     \
        --resume_path            "./logs_NCMAPSS/${SRC}_${TGT}/${model}/final_s42/" \
        --warm_start_checkpoint  "./logs_NCMAPSS/${SRC}_${TGT}/${model}/final_s42/best_checkpoint.pth" \
        --save_path              "$tag"    \
        --logs_root              ./logs_NCMAPSS \
        --info                   "${model}_${SRC}_${TGT}_seed${SEED}_final" \
        || echo "✗  [FAIL] $SRC → $TGT | $model | seed $SEED"

done; done; done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SRC $SRC complete.  Results in: logs_NCMAPSS/"
echo "════════════════════════════════════════════════════════════════"
