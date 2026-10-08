#!/usr/bin/env bash
# =============================================================================
# Full LR sweep -- all DA models, SOURCE=FD001
#
# All hyperparameters except --learning_rate from configs/hparams_tune.py.
# NDC_PDMN runs twice: hidden_dim=32 and hidden_dim=64.
#
# Usage:
#   bash scripts/grid_search_all_01.sh                  # all models, all targets
#   bash scripts/grid_search_all_01.sh PDMN             # one model
#   bash scripts/grid_search_all_01.sh OCS_DANN FD002   # one model, one target
#
# Save output:
#   nohup bash scripts/grid_search_all_01.sh > all_FD001.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="tune_hparams.py"
SOURCE="FD002"

ALL_MODELS=(
    # Ours
    "PDMN"
    "LBGN_RUL"
    "NDC_PDMN"
)
ALL_TARGETS=("FD002" "FD003" "FD004")

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
NDC_HIDDEN_DIMS=(32)

# Count: NDC_PDMN counts double
n_active=0
for m in "${ALL_MODELS[@]}"; do
    has_model "$m" || continue
    if [[ "$m" == "NDC_PDMN" ]]; then
        n_active=$(( n_active + ${#NDC_HIDDEN_DIMS[@]} ))
    else
        n_active=$(( n_active + 1 ))
    fi
done
total=$(( n_active * ${#TARGETS[@]} * ${#LR_VALUES[@]} ))
done_count=0

echo "=================================================================="
echo "  Full LR sweep  |  SOURCE=$SOURCE  |  $total experiments"
echo "  Models  : ${MODELS[*]}"
echo "  Targets : ${TARGETS[*]}"
echo "  LR      : ${LR_VALUES[*]}"
echo "  NDC_PDMN hidden_dim : ${NDC_HIDDEN_DIMS[*]}"
echo "=================================================================="

run_exp() {
    local model="$1" tgt="$2" lr="$3" tag="$4"
    shift 4
    done_count=$(( done_count + 1 ))
    echo ""
    echo ">  [$done_count/$total] $model | $SOURCE -> $tgt | $tag"
    $PY "$SCRIPT"         --model_name          "$model"          --dataset_name        CMAPSS            --Data_id_CMAPSS      "$SOURCE"         --Data_id_CMAPSS_test "$tgt"            --info                "$tag"            --learning_rate       "$lr"             "$@"
}

for tgt in "${TARGETS[@]}"; do
for lr  in "${LR_VALUES[@]}"; do

    has_model working_model_RUL     && run_exp working_model_RUL     "$tgt" "$lr" "lr${lr}"
    has_model PDMN                  && run_exp PDMN                  "$tgt" "$lr" "lr${lr}"
    has_model LBGN_RUL && run_exp LBGN_RUL "$tgt" "$lr" "lr${lr}"

    if has_model NDC_PDMN; then
        for hd in "${NDC_HIDDEN_DIMS[@]}"; do
            run_exp NDC_PDMN "$tgt" "$lr" "hd${hd}_lr${lr}" --hidden_dim "$hd"
        done
    fi

    has_model DAGCN_RUL             && run_exp DAGCN_RUL             "$tgt" "$lr" "lr${lr}"
    has_model EviAdaptRUL           && run_exp EviAdaptRUL           "$tgt" "$lr" "lr${lr}"
    has_model DAST_RUL              && run_exp DAST_RUL              "$tgt" "$lr" "lr${lr}"
    has_model CADA_RUL              && run_exp CADA_RUL              "$tgt" "$lr" "lr${lr}"
    has_model TACDA_RUL             && run_exp TACDA_RUL             "$tgt" "$lr" "lr${lr}"

    has_model OCS_DANN              && run_exp OCS_DANN              "$tgt" "$lr" "lr${lr}"
    has_model MDAN_RUL              && run_exp MDAN_RUL              "$tgt" "$lr" "lr${lr}"
    has_model CCDG_RUL              && run_exp CCDG_RUL              "$tgt" "$lr" "lr${lr}"

done; done

echo ""
echo "=================================================================="
echo "  Done.  $done_count / $total experiments.  Results in: logs/"
echo "=================================================================="
