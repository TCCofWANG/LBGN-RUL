#!/usr/bin/env bash
# =============================================================================
# Smoke-test: verify every model imports, builds, and completes 1 training
# epoch without error.
#
# Runs each model for train_epochs=2 on FD001->FD002 (small cross-domain pair).
# Two-phase models (CADA, TACDA, EviAdapt, MDAN) also get pretrain_epochs=1
# so the phase transition is exercised within the 2 epochs.
#
# Exit codes per model are collected; a summary is printed at the end.
#
# Usage:
#   bash scripts/check_models.sh            # all models
#   bash scripts/check_models.sh NDC_PDMN   # one model
# =============================================================================

# do NOT set -e — we want to continue after failures
set -uo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="tune_hparams.py"
SRC="FD001"
TGT="FD002"
EPOCHS=2

# ── Step 0: fast import check (no data loading) ───────────────────────────────
echo ""
echo ">> Step 0: checking all model imports ..."
if ! $PY scripts/check_imports.py; then
    echo ""
    echo "  Import check FAILED -- fix missing/broken files before training."
    exit 1
fi
echo ""

ALL_MODELS=(
    "working_model_RUL"
    "PDMN"
    "LBGN_RUL"
    "NDC_PDMN"
    "DAGCN_RUL"
    "EviAdaptRUL"
    "DAST_RUL"
    "CADA_RUL"
    "TACDA_RUL"
    "OCS_DANN"
    "MDAN_RUL"
    "CCDG_RUL"
    "AIDGN"
    "DLGNet"
)

# Filter to CLI args if provided
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# Accumulate results
PASS=()
FAIL=()
FAIL_MSGS=()

echo "=================================================================="
echo "  Model smoke-test  |  $SRC -> $TGT  |  ${#MODELS[@]} models"
echo "=================================================================="

run_check() {
    local model="$1"
    shift   # remaining: extra flags specific to this model

    echo ""
    echo ">> Checking $model ..."

    # Capture stdout+stderr so failures don't spam the terminal
    local logfile
    logfile=$(mktemp)

    $PY "$SCRIPT" \
        --model_name          "$model"   \
        --dataset_name        CMAPSS     \
        --Data_id_CMAPSS      "$SRC"     \
        --Data_id_CMAPSS_test "$TGT"     \
        --train_epochs        "$EPOCHS"  \
        --info                "smoketest" \
        "$@" \
        > "$logfile" 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "   PASS  $model"
        PASS+=("$model")
    else
        local last_err
        last_err=$(tail -5 "$logfile" | tr '\n' ' ')
        echo "   FAIL  $model"
        echo "   Last output: $last_err"
        FAIL+=("$model")
        FAIL_MSGS+=("$model: $last_err")
    fi
    rm -f "$logfile"
}

has_model() { [[ " ${MODELS[*]} " == *" $1 "* ]]; }

# ── Ours ──────────────────────────────────────────────────────────────────────
has_model working_model_RUL     && run_check working_model_RUL
has_model PDMN                  && run_check PDMN
has_model LBGN_RUL && run_check LBGN_RUL
has_model NDC_PDMN              && run_check NDC_PDMN --hidden_dim 64

# ── Prior DA comparisons ──────────────────────────────────────────────────────
has_model DAGCN_RUL  && run_check DAGCN_RUL
has_model EviAdaptRUL && run_check EviAdaptRUL \
    --pretrain_epochs 1
has_model DAST_RUL   && run_check DAST_RUL
has_model CADA_RUL   && run_check CADA_RUL \
    --pretrain_epochs 1
has_model TACDA_RUL  && run_check TACDA_RUL \
    --pretrain_epochs 1

# ── New baselines ─────────────────────────────────────────────────────────────
has_model OCS_DANN  && run_check OCS_DANN
has_model MDAN_RUL  && run_check MDAN_RUL \
    --pretrain_epochs 1
has_model CCDG_RUL  && run_check CCDG_RUL

# ── Non-DA prior models (plain training(), cross-domain eval via test_Data_id) ──
has_model AIDGN     && run_check AIDGN
has_model DLGNet    && run_check DLGNet

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  RESULTS: ${#PASS[@]} passed / ${#FAIL[@]} failed"
echo "=================================================================="

if [[ ${#PASS[@]} -gt 0 ]]; then
    echo "  PASS:"
    for m in "${PASS[@]}"; do echo "    [OK]  $m"; done
fi

if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo "  FAIL:"
    for msg in "${FAIL_MSGS[@]}"; do echo "    [!!]  $msg"; done
    echo ""
    echo "  Fix the above, then re-run:"
    echo "    bash scripts/check_models.sh ${FAIL[*]}"
    exit 1
fi

echo ""
echo "  All models OK -- safe to launch grid search."
exit 0
