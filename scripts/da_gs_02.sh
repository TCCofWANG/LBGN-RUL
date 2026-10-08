#!/usr/bin/env bash
# =============================================================================
# Domain Adaptation Sweep -- LBGN_RUL (Seeds, lam_mar, lam_da), SOURCE=FD001
#
# LR is NOT swept here -- it just uses whatever configs/hparams.py has tuned
# for the given source dataset (no --learning_rate flag passed), so this
# isolates lam_mar/lam_da. Range widened past the old 0.001-0.5 grid because
# that range showed almost no effect on RMSE/score (mar_loss's margin bug
# meant lam_mar did nothing at all until fixed; lam_da needed to go well
# above 0.5 before an effect showed up at all on FD001->FD002, e.g.
# lam_da=1.0/lam_mar=1.5 vs the old 0.01/0.05 defaults).
#
# Usage:
#   bash scripts/da_gs_01.sh                  # default LBGN_RUL, all targets
#   bash scripts/da_gs_01.sh LBGN_RUL FD002   # target filter (e.g. just the
#                                              # pair where an effect showed up)
#
# Save output:
#   nohup bash scripts/da_gs_01.sh > all_FD001.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

ALL_MODELS=("LBGN_RUL")
ALL_TARGETS=("FD001" "FD003" "FD004")

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
SOURCE="FD002"

# -- Parse optional CLI filters
MODELS=()
TARGETS=()
for arg in "$@"; do
    [[ "$arg" == FD* ]] && TARGETS+=("$arg") || MODELS+=("$arg")
done
[[ ${#MODELS[@]}  -eq 0 ]] && MODELS=("${ALL_MODELS[@]}")
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("${ALL_TARGETS[@]}")

has_model() { [[ " ${MODELS[*]} " == *" $1 "* ]]; }

SEEDS=(64 123)
LAM_MAR_VALUES=(0.01 0.5 1.0 1.5 2.0)
LAM_DA_VALUES=(0.1 0.5 1.0 1.5)

# Count total active runs
n_active=0
if has_model LBGN_RUL; then
    n_active=$(( ${#TARGETS[@]} * ${#SEEDS[@]} * ${#LAM_MAR_VALUES[@]} * ${#LAM_DA_VALUES[@]} ))
fi
total=$n_active
done_count=0

echo "=================================================================="
echo "  Hyperparameter Sweep  |  SOURCE=$SOURCE  |  $total experiments"
echo "  Models      : ${MODELS[*]}"
echo "  Targets     : ${TARGETS[*]}"
echo "  LR          : not swept -- uses configs/hparams.py's tuned default"
echo "  Seeds       : ${SEEDS[*]}"
echo "  lam_mar     : ${LAM_MAR_VALUES[*]}"
echo "  lam_da      : ${LAM_DA_VALUES[*]}"
echo "=================================================================="

run_exp() {
    local model="$1" tgt="$2" seed="$3" lmar="$4" lda="$5" tag="$6"
    shift 6
    done_count=$(( done_count + 1 ))
    echo ""
    echo "> [$done_count/$total] $model | $SOURCE -> $tgt | Seed=$seed | lam_mar=$lmar | lam_da=$lda"
    $PY "$SCRIPT" \
        --model_name          "$model" \
        --dataset_name        CMAPSS \
        --Data_id_CMAPSS      "$SOURCE" \
        --Data_id_CMAPSS_test "$tgt" \
        --loss_type           DA \
        --seed                "$seed" \
        --lam_mar             "$lmar" \
        --lam_da              "$lda" \
        --info                "$tag" \
        "$@"
}

if has_model LBGN_RUL; then
    for seed in "${SEEDS[@]}"; do
        for tgt in "${TARGETS[@]}"; do
            for lmar in "${LAM_MAR_VALUES[@]}"; do
                for lda in "${LAM_DA_VALUES[@]}"; do
                    run_exp LBGN_RUL "$tgt" "$seed" "$lmar" "$lda" "s${seed}_lmar${lmar}_lda${lda}"
                done
            done
        done
    done
fi

echo ""
echo "=================================================================="
echo "  Done.  $done_count / $total experiments.  Results in: logs/"
echo "=================================================================="
