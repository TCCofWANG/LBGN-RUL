#!/usr/bin/env bash
# =============================================================================
# Warm-start further training from the current best checkpoint per (pair,
# model) in ./logs_best/ (built by scripts/copy_best_checkpoints.py), instead
# of training from scratch -- new --seed values only affect things AFTER
# weight loading (data shuffling order, dropout masks, optimizer state),
# not the initial weights.
#
# Skips ablation checkpoints for now (only the plain {model}/exp0 folders,
# not {model}_ablation/exp0).
#
# Output goes to a NEW tag (warmstart_s{seed}) under the normal ./logs tree,
# distinct from final_s{seed} (the original 5-seed campaign) so it can never
# collide with or overwrite that data. info is tagged "_warmstart" (not
# "_final"), so these rows are automatically excluded from every existing
# "_final"-filtering analysis script (final_table.py, add_pdmn_ndc_to_xlsx.py,
# copy_best_checkpoints.py) unless you deliberately opt them in later.
#
# Resume-safe: skips a (pair, model, seed) combo if its warmstart_s{seed}
# checkpoint already exists.
#
# Usage:
#   bash scripts/run_warmstart_from_best.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
LOGS_BEST_DIR="./logs_best"

MODELS=("LBGN_RUL" "PDMN" "NDC_PDMN" "DAGCN_RUL" "EviAdaptRUL" "DAST_RUL" "CADA_RUL" "TACDA_RUL" "OCS_DANN" "MDAN_RUL" "CCDG_RUL" "AIDGN" "DLGNet")

# New seeds, deliberately disjoint from the original campaign's (0 42 64 7 123)
# so warm-start runs can never be confused with from-scratch ones.
SEEDS=(1 64 1001 1002)

PAIRS=(
    "FD003 FD001" "FD003 FD002" "FD003 FD004"
    "FD004 FD001" "FD004 FD002" "FD004 FD003"
)

total=$(( ${#PAIRS[@]} * ${#MODELS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  Warm-start from logs_best  |  $total experiments"
echo "  Models : ${MODELS[*]}"
echo "  Seeds  : ${SEEDS[*]}"
echo "══════════════════════════════════════════════════════════════"

for pair in "${PAIRS[@]}"; do
    read -r SRC tgt <<< "$pair"
for model in "${MODELS[@]}"; do

    best_ckpt="${LOGS_BEST_DIR}/${SRC}_${tgt}/${model}/exp0/best_checkpoint.pth"
    best_hparam_yaml="${LOGS_BEST_DIR}/${SRC}_${tgt}/${model}/exp0/hparam.yaml"
    if [[ ! -f "$best_ckpt" ]]; then
        echo "⚠  no best checkpoint for $SRC → $tgt | $model at $best_ckpt -- skipping all seeds"
        done_count=$(( done_count + ${#SEEDS[@]} ))
        continue
    fi

for seed in "${SEEDS[@]}"; do

    done_count=$(( done_count + 1 ))
    tag="warmstart_s${seed}"
    ckpt="./logs/${SRC}_${tgt}/${model}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $tgt | $model | seed $seed (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $tgt | $model | seed $seed  (warm-start from $best_ckpt)"

    $PY "$SCRIPT" \
        --model_name              "$model"     \
        --dataset_name             CMAPSS      \
        --Data_id_CMAPSS           "$SRC"      \
        --Data_id_CMAPSS_test      "$tgt"      \
        --seed                     "$seed"     \
        --train                    True        \
        --resume                   True        \
        --warm_start_checkpoint    "$best_ckpt" \
        --warm_start_hparam_yaml   "$best_hparam_yaml" \
        --save_path                "$tag"      \
        --info                     "${model}_${SRC}_${tgt}_seed${seed}_warmstart" \
        || echo "✗  [FAIL] $SRC → $tgt | $model | seed $seed"

done; done; done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Warm-start runs complete.  Results in: logs/  (info tag: _warmstart)"
echo "════════════════════════════════════════════════════════════════"
