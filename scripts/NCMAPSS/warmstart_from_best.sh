#!/usr/bin/env bash
# =============================================================================
# Warm-start further training from the best checkpoint per (pair, model) in
# ./ncmapss_logs_best/ (built by scripts/copy_best_checkpoints.py --dataset_name
# N_CMAPSS against the grid-search pool in ./ncmapss_lr_logs/), instead of
# training from scratch again -- new --seed values only affect things AFTER
# weight loading (data shuffling order, dropout masks, optimizer state), not
# the initial weights.
#
# LBGN_RUL is treated exactly like every baseline here (no cold-start
# exception) -- all models in MODELS warm-start the same way, giving mean+-std
# over the SEEDS below for every (pair, model).
#
# Output goes under ./logs_NCMAPSS/ tagged warmstart_s{seed}, distinct from
# ./ncmapss_lr_logs/ (the grid search, its own dedicated root) and any future
# final_s{seed} campaign (run_ncmapss_da.sh, same ./logs_NCMAPSS/ root but a
# different tag), so nothing can collide. Resume-safe: skips a (pair, model,
# seed) combo if its warmstart_s{seed} checkpoint already exists.
#
# Usage:
#   bash scripts/NCMAPSS/warmstart_from_best.sh
#   bash scripts/NCMAPSS/warmstart_from_best.sh LBGN_RUL   # one model
#   bash scripts/NCMAPSS/warmstart_from_best.sh DS02       # one source (all its targets)
#
# Prereqs:
#   1. scripts/NCMAPSS/grid_search_lr_seed.sh has produced ./ncmapss_lr_logs/
#   2. python scripts/copy_best_checkpoints.py --logs_dir ./ncmapss_lr_logs \
#          --dataset_name N_CMAPSS --out ./ncmapss_logs_best
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
LOGS_BEST_DIR="./ncmapss_logs_best"

ALL_MODELS=("LBGN_RUL" "PDMN" "NDC_PDMN" "DAGCN_RUL" "EviAdaptRUL" "DAST_RUL" "CADA_RUL" "TACDA_RUL" "OCS_DANN" "MDAN_RUL" "CCDG_RUL" "AIDGN" "DLGNet")
ALL_SOURCES=("DS01" "DS02" "DS03" "DS04")

# New seeds, deliberately disjoint from the grid search's (0) so
# warm-start runs can never be confused with the grid-search pool. 5 seeds
# gives mean+-std over 5 runs; add more (e.g. 1003 1004 1005) to extend to 10.
SEEDS=(1 7 41 1001 1002)

# -- Parse optional CLI filters (model names vs DS-prefixed source filters)
MODELS=()
SOURCES=()
for arg in "$@"; do
    [[ "$arg" == DS* ]] && SOURCES+=("$arg") || MODELS+=("$arg")
done
[[ ${#MODELS[@]}  -eq 0 ]] && MODELS=("${ALL_MODELS[@]}")
[[ ${#SOURCES[@]} -eq 0 ]] && SOURCES=("${ALL_SOURCES[@]}")

has_model() { [[ " ${MODELS[*]} " == *" $1 "* ]]; }

total=0
for src in "${SOURCES[@]}"; do
    for tgt in "${ALL_SOURCES[@]}"; do
        [[ "$tgt" == "$src" ]] && continue
        total=$(( total + ${#MODELS[@]} * ${#SEEDS[@]} ))
    done
done
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  N-CMAPSS warm-start from $LOGS_BEST_DIR  |  $total experiments"
echo "  Models  : ${MODELS[*]}"
echo "  Sources : ${SOURCES[*]}  (targets = other DS01-04 members)"
echo "  Seeds   : ${SEEDS[*]}"
echo "══════════════════════════════════════════════════════════════"

for src in "${SOURCES[@]}"; do
for tgt in "${ALL_SOURCES[@]}"; do
    [[ "$tgt" == "$src" ]] && continue
for model in "${MODELS[@]}"; do

    # copy_best_checkpoints.py names this folder seed{N} for whichever seed
    # actually won (not a fixed "exp0"), and guarantees at most one such
    # folder exists per (pair, model) -- so a glob here is always unambiguous.
    best_ckpt=$(ls "${LOGS_BEST_DIR}/${src}_${tgt}/${model}"/seed*/best_checkpoint.pth 2>/dev/null | head -n1)
    if [[ -z "$best_ckpt" ]]; then
        echo "⚠  no best checkpoint for $src → $tgt | $model under ${LOGS_BEST_DIR}/${src}_${tgt}/${model}/ -- skipping all seeds"
        done_count=$(( done_count + ${#SEEDS[@]} ))
        continue
    fi
    best_hparam_yaml="$(dirname "$best_ckpt")/hparam.yaml"

for seed in "${SEEDS[@]}"; do

    done_count=$(( done_count + 1 ))
    tag="warmstart_s${seed}"
    ckpt="./logs_NCMAPSS/${src}_${tgt}/${model}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $src → $tgt | $model | seed $seed (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $src → $tgt | $model | seed $seed  (warm-start from $best_ckpt)"

    $PY "$SCRIPT" \
        --model_name              "$model"    \
        --dataset_name            N_CMAPSS    \
        --Data_id_N_CMAPSS        "$src"      \
        --Data_id_N_CMAPSS_test   "$tgt"      \
        --seed                    "$seed"     \
        --train                   True        \
        --resume                  True        \
        --warm_start_checkpoint   "$best_ckpt" \
        --warm_start_hparam_yaml  "$best_hparam_yaml" \
        --save_path               "$tag"      \
        --logs_root               ./logs_NCMAPSS \
        --info                    "${model}_${src}_${tgt}_seed${seed}_warmstart" \
        || echo "✗  [FAIL] $src → $tgt | $model | seed $seed"

done; done; done; done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Warm-start runs complete.  Results in: logs_NCMAPSS/  (info tag: _warmstart)"
echo "  Compute mean+-std per (pair, model) over the warmstart_s{seed} rows."
echo "══════════════════════════════════════════════════════════════"
