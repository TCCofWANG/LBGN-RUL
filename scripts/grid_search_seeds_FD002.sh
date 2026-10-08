#!/usr/bin/env bash
# =============================================================================
# LR x seed stability sweep -- LBGN_RUL, PDMN, NDC_PDMN only
# SOURCE=FD001
#
# Scoped-down version of grid_search_all_01.sh: instead of a single-seed LR
# sweep across all models, this repeats the LR sweep at the two seeds that
# produced catastrophic score blow-ups in the 5-seed final campaign --
# seed=123 (worst overall: PDMN FD002->FD003 score 19854, FiLMHeavy
# FD001->FD004 score 17115) and seed=64 (second-worst: PDMN FD002->FD003
# score 11065, FiLMHeavy FD002->FD004/FD002->FD003 scores 9714/7363).
# See analysis/final_table.py output for the full ranking this was drawn from.
#
# Goal: find out whether a lower/more-conservative LR (matching the fix
# already validated for FiLMHeavy/FD001, see scripts/diagnose_fd001_stability.sh)
# also tames these seeds for PDMN and NDC_PDMN, the same way it did for
# FiLMHeavy -- before deciding whether to touch their hparams.py entries.
#
# All hyperparameters except --learning_rate/--seed from configs/hparams_tune.py.
# NDC_PDMN runs twice per (lr, seed): hidden_dim=32 and hidden_dim=64.
#
# Usage:
#   bash scripts/grid_search_seeds_FD002.sh                  # all 3 models, all targets
#   bash scripts/grid_search_seeds_FD002.sh PDMN              # one model
#   bash scripts/grid_search_seeds_FD002.sh PDMN FD003         # one model, one target
#
# Save output:
#   nohup bash scripts/grid_search_seeds_FD002.sh > seeds_FD002.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="tune_hparams.py"
SOURCE="FD002"

ALL_MODELS=("LBGN_RUL" "PDMN" "NDC_PDMN")
ALL_TARGETS=("FD001" "FD003" "FD004")

# -- Parse optional CLI filters
MODELS=()
TARGETS=()
for arg in "$@"; do
    [[ "$arg" == FD* ]] && TARGETS+=("$arg") || MODELS+=("$arg")
done
[[ ${#MODELS[@]}  -eq 0 ]] && MODELS=("${ALL_MODELS[@]}")
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("${ALL_TARGETS[@]}")

has_model() { [[ " ${MODELS[*]} " == *" $1 "* ]]; }

LR_VALUES=(0.005 0.001 0.0005)
SEEDS=(123 64)
NDC_HIDDEN_DIMS=(32 64)

n_active=0
for m in "${ALL_MODELS[@]}"; do
    has_model "$m" || continue
    if [[ "$m" == "NDC_PDMN" ]]; then
        n_active=$(( n_active + ${#NDC_HIDDEN_DIMS[@]} ))
    else
        n_active=$(( n_active + 1 ))
    fi
done
total=$(( n_active * ${#TARGETS[@]} * ${#LR_VALUES[@]} * ${#SEEDS[@]} ))
done_count=0

echo "=================================================================="
echo "  LR x seed stability sweep  |  SOURCE=$SOURCE  |  $total experiments"
echo "  Models  : ${MODELS[*]}"
echo "  Targets : ${TARGETS[*]}"
echo "  LR      : ${LR_VALUES[*]}"
echo "  Seeds   : ${SEEDS[*]}  (catastrophic seeds from the 5-seed campaign)"
echo "  NDC_PDMN hidden_dim : ${NDC_HIDDEN_DIMS[*]}"
echo "=================================================================="

run_exp() {
    local model="$1" tgt="$2" lr="$3" seed="$4" tag="$5"
    shift 5
    done_count=$(( done_count + 1 ))
    echo ""
    echo ">  [$done_count/$total] $model | $SOURCE -> $tgt | $tag"
    $PY "$SCRIPT" \
        --model_name          "$model"   \
        --dataset_name        CMAPSS     \
        --Data_id_CMAPSS      "$SOURCE"  \
        --Data_id_CMAPSS_test "$tgt"     \
        --info                "$tag"     \
        --logs_root           ./lr_logs  \
        --learning_rate       "$lr"      \
        --seed                "$seed"    \
        "$@"
}

for tgt  in "${TARGETS[@]}"; do
for seed in "${SEEDS[@]}"; do
for lr   in "${LR_VALUES[@]}"; do

    has_model LBGN_RUL && run_exp LBGN_RUL "$tgt" "$lr" "$seed" "lr${lr}_seed${seed}"
    has_model PDMN                  && run_exp PDMN                  "$tgt" "$lr" "$seed" "lr${lr}_seed${seed}"

    if has_model NDC_PDMN; then
        for hd in "${NDC_HIDDEN_DIMS[@]}"; do
            run_exp NDC_PDMN "$tgt" "$lr" "$seed" "hd${hd}_lr${lr}_seed${seed}" --hidden_dim "$hd"
        done
    fi

done; done; done

echo ""
echo "=================================================================="
echo "  Done.  $done_count / $total experiments.  Results in: logs/"
echo "=================================================================="
