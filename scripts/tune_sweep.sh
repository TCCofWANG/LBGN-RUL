#!/usr/bin/env bash
# =============================================================================
# Hyperparameter sweep for DA_RUL models
#
# Strategy: sequential (not full grid) — one param at a time.
# WORKFLOW:
#   Step 1 — run LR sweep:   bash tune_sweep.sh working_model_RUL
#   Step 2 — read CSV logs, fill in BEST_LR_* below with the best value
#   Step 3 — comment out step 1 block, re-run to sweep remaining params
#
# Usage:
#   bash scripts/tune_sweep.sh                   # all models
#   bash scripts/tune_sweep.sh working_model_RUL # one model
#
# Requirements: conda/venv with project dependencies activated.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
SCRIPT="tune_hparams.py"
LOG_DIR="logs/tuning"
mkdir -p "$LOG_DIR"

SOURCE="FD001"
TARGETS=("FD002" "FD003" "FD004")

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("working_model_RUL" "PDMN" "LBGN_RUL"
            "DAGCN_RUL" "EviAdaptRUL" "DAST_RUL" "CADA_RUL" "TACDA_RUL")
fi

# =============================================================================
# ★  FILL IN BEST VALUES after each LR sweep, before running the next steps  ★
# =============================================================================
BEST_LR_working_model_RUL=5e-3   # best from lr sweep
BEST_LR_PDMN=1e-3
BEST_LR_LBGN_RUL=5e-3
BEST_LR_DAGCN_RUL=1e-3
BEST_LR_EviAdaptRUL=1e-3
BEST_LR_DAST_RUL=1e-3
BEST_LR_CADA_RUL=1e-3
BEST_LR_TACDA_RUL=1e-3
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one experiment, log stdout+stderr, skip if already done
# ─────────────────────────────────────────────────────────────────────────────
run_exp() {
    local model="$1"; shift            # first arg = model name
    local tgt="$1";   shift            # second arg = target domain
    local tag="$1";   shift            # third arg = human-readable tag
    local extra=("$@")                 # remaining = extra --key value pairs

    local log_file="$LOG_DIR/${model}/${SOURCE}x${tgt}_${tag}.log"
    mkdir -p "$LOG_DIR/${model}"

    echo ""
    echo "▶  $model | $SOURCE → $tgt | $tag"

    $PY "$SCRIPT" \
        --model_name          "$model"  \
        --dataset_name        CMAPSS    \
        --Data_id_CMAPSS      "$SOURCE" \
        --Data_id_CMAPSS_test "$tgt"    \
        --info                "$tag"    \
        "${extra[@]}"                   \
        2>&1 | tee "$log_file"

    echo "   ✓ logged → $log_file"
}

# ─────────────────────────────────────────────────────────────────────────────
# working_model_RUL — OCC-weighted MSE + MAR + adjacency-MMD
#   Key params: learning_rate, lam_mar, lam_da, LCE_dim, n_bands, hop, input_length
# ─────────────────────────────────────────────────────────────────────────────
if [[ " ${MODELS[*]} " == *" working_model_RUL "* ]]; then
    echo ""; echo "══════════════ working_model_RUL ══════════════"

    # 1. learning_rate sweep (all other params stay at hparam defaults)
    for lr in 5e-3 1e-2; do
        for tgt in "${TARGETS[@]}"; do
            run_exp working_model_RUL "$tgt" "lr${lr}" \
                --learning_rate "$lr"
        done
    done

    LR=$BEST_LR_working_model_RUL

    # 2. lam_mar sweep
    for lam in 0.01 0.05 0.1 0.5; do
        for tgt in "${TARGETS[@]}"; do
            run_exp working_model_RUL "$tgt" "lam_mar${lam}" \
                --learning_rate "$LR" --lam_mar "$lam"
        done
    done

    # 3. lam_da sweep
    for lam in 0.05 0.1 0.2 0.5; do
        for tgt in "${TARGETS[@]}"; do
            run_exp working_model_RUL "$tgt" "lam_da${lam}" \
                --learning_rate "$LR" --lam_da "$lam"
        done
    done

    # 4. LCE_dim sweep
    for dim in 8 16 32; do
        for tgt in "${TARGETS[@]}"; do
            run_exp working_model_RUL "$tgt" "LCE_dim${dim}" \
                --learning_rate "$LR" --LCE_dim "$dim"
        done
    done

    # 5. n_bands sweep
    for nb in 2 4 8; do
        for tgt in "${TARGETS[@]}"; do
            run_exp working_model_RUL "$tgt" "n_bands${nb}" \
                --learning_rate "$LR" --n_bands "$nb"
        done
    done

    # 6. hop sweep
    for hop in 1 2 3; do
        for tgt in "${TARGETS[@]}"; do
            run_exp working_model_RUL "$tgt" "hop${hop}" \
                --learning_rate "$LR" --hop "$hop"
        done
    done

    # 7. window size sweep
    for ws in 30 35 40 45 50; do
        for tgt in "${TARGETS[@]}"; do
            run_exp working_model_RUL "$tgt" "ws${ws}" \
                --learning_rate "$LR" --input_length "$ws"
        done
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# PDMN — plain MSE + routing-MMD + struct loss
#   Key params: learning_rate, lam_struct, lam_rmmd, K, dropout
# ─────────────────────────────────────────────────────────────────────────────
if [[ " ${MODELS[*]} " == *" PDMN "* ]]; then
    echo ""; echo "══════════════ PDMN ══════════════"

    # 1. learning_rate
    for lr in 5e-3 1e-2; do
        for tgt in "${TARGETS[@]}"; do
            run_exp PDMN "$tgt" "lr${lr}" \
                --learning_rate "$lr"
        done
    done

    LR=$BEST_LR_PDMN

    # 2. lam_rmmd  (highest-leverage PDMN param)
    for lam in 0.05 0.1 0.2 1.0; do
        for tgt in "${TARGETS[@]}"; do
            run_exp PDMN "$tgt" "lam_rmmd${lam}" \
                --learning_rate "$LR" --lam_rmmd "$lam"
        done
    done

    # 3. lam_struct
    for lam in 0.01 0.1 0.5; do
        for tgt in "${TARGETS[@]}"; do
            run_exp PDMN "$tgt" "lam_struct${lam}" \
                --learning_rate "$LR" --lam_struct "$lam"
        done
    done

    # 4. K (routing iterations)
    for k in 4 8 16; do
        for tgt in "${TARGETS[@]}"; do
            run_exp PDMN "$tgt" "K${k}" \
                --learning_rate "$LR" --K "$k"
        done
    done

    # 5. dropout
    for dp in 0.1 0.2 0.3 0.5; do
        for tgt in "${TARGETS[@]}"; do
            run_exp PDMN "$tgt" "dropout${dp}" \
                --learning_rate "$LR" --dropout "$dp"
        done
    done

    # 6. window size sweep
    for ws in 30 35 40 45 50; do
        for tgt in "${TARGETS[@]}"; do
            run_exp PDMN "$tgt" "ws${ws}" \
                --learning_rate "$LR" --input_length "$ws"
        done
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# LBGN_RUL — ablation of working_model_RUL (depth-wise FiLM)
#   Key params: learning_rate, LCE_dim, hop, n_bands, lam_mar
#   (same loss as working_model_RUL; focus on architecture params)
# ─────────────────────────────────────────────────────────────────────────────
if [[ " ${MODELS[*]} " == *" LBGN_RUL "* ]]; then
    echo ""; echo "══════════════ LBGN_RUL ══════════════"

    # 1. learning_rate
    for lr in 5e-3 1e-2; do
        for tgt in "${TARGETS[@]}"; do
            run_exp LBGN_RUL "$tgt" "lr${lr}" \
                --learning_rate "$lr"
        done
    done

    LR=$BEST_LR_LBGN_RUL

    # 2. LCE_dim  (controls richness of per-depth FiLM conditioning)
    for dim in 8 16 32; do
        for tgt in "${TARGETS[@]}"; do
            run_exp LBGN_RUL "$tgt" "LCE_dim${dim}" \
                --learning_rate "$LR" --LCE_dim "$dim"
        done
    done

    # 3. hop  (adds more independent FiLM layers per depth)
    for hop in 1 2 3; do
        for tgt in "${TARGETS[@]}"; do
            run_exp LBGN_RUL "$tgt" "hop${hop}" \
                --learning_rate "$LR" --hop "$hop"
        done
    done

    # 4. n_bands
    for nb in 2 4 8; do
        for tgt in "${TARGETS[@]}"; do
            run_exp LBGN_RUL "$tgt" "n_bands${nb}" \
                --learning_rate "$LR" --n_bands "$nb"
        done
    done

    # 5. lam_mar
    for lam in 0.01 0.05 0.1; do
        for tgt in "${TARGETS[@]}"; do
            run_exp LBGN_RUL "$tgt" "lam_mar${lam}" \
                --learning_rate "$LR" --lam_mar "$lam"
        done
    done

    # 6. window size sweep
    for ws in 30 35 40 45 50; do
        for tgt in "${TARGETS[@]}"; do
            run_exp LBGN_RUL "$tgt" "ws${ws}" \
                --learning_rate "$LR" --input_length "$ws"
        done
    done
fi

# =============================================================================
# BRUTE-FORCE GRID — working_model_RUL, LBGN_RUL, PDMN
#
# Full grid over (lr × lam_mar × dropout) for every source→target pair.
# Run independently:  bash scripts/tune_sweep.sh --grid
# =============================================================================
if [[ " $* " == *" --grid "* ]]; then
    echo ""; echo "══════════════ BRUTE-FORCE GRID ══════════════"

    GRID_MODELS=("working_model_RUL" "LBGN_RUL" "PDMN")
    GRID_TARGETS=("FD001" "FD002" "FD003" "FD004")   # includes same-domain FD001

    for model in "${GRID_MODELS[@]}"; do
        echo ""; echo "── $model ──"
        for tgt in "${GRID_TARGETS[@]}"; do
            for lr in 0.005 0.001 0.0005 0.0001; do
                for lam in 0.05 0.01 0.005; do
                    for dp in 0.1 0.2 0.2; do
                        tag="lr${lr}_lam${lam}_dp${dp}"
                        local_log="$LOG_DIR/${model}/${SOURCE}x${tgt}_grid_${tag}.log"
                        mkdir -p "$LOG_DIR/${model}"
                        echo "▶  $model | $SOURCE → $tgt | $tag"
                        $PY "$SCRIPT" \
                            --model_name          "$model"   \
                            --dataset_name        CMAPSS     \
                            --Data_id_CMAPSS      "$SOURCE"  \
                            --Data_id_CMAPSS_test "$tgt"     \
                            --info                "grid_${tag}" \
                            --learning_rate       "$lr"      \
                            --lam_mar             "$lam"     \
                            --dropout             "$dp"      \
                            2>&1 | tee "$local_log"
                        echo "   ✓ logged → $local_log"
                    done
                done
            done
        done
    done
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  Sweep complete. Results in: tune_out/"
echo "  Logs in:                    $LOG_DIR"
echo "════════════════════════════════════════════════════"
