#!/usr/bin/env bash
# =============================================================================
# LBGN_RUL lam_mar/lam_da sensitivity analysis -- CMAPSS, all 12 pairs.
#
# WHY THIS SCRIPT (not da_gs_01-04.sh)
# -------------------------------------
# da_gs_01-04.sh sweep the full 5x4 lam_mar x lam_da cross-product x 2 seeds
# x 3 targets, PER SOURCE (4 separate scripts) -- 480 runs total for CMAPSS.
# Their output CSVs no longer exist on disk (confirmed by repo audit), and
# configs/hparams.py's "new_loss_check grid" comments citing specific
# percentages from that sweep are therefore NOT independently verifiable
# from any surviving data.
#
# This script instead runs a SMALL, FIXED set of (lam_mar, lam_da) points --
# mirroring EviAdaptRUL's own sensitivity analysis, which tests a handful of
# discrete configs (not a full grid) across every cross-domain pair -- at
# seed 42 only, consolidated across all 4 sources into one script covering
# all 12 pairs.
#
# THE 5 CHOSEN (lam_mar, lam_da) POINTS -- v2 (redesigned)
# ---------------------------------------------------------
# The original 5-point design (see git history / prior comment block) chose
# points to match each CMAPSS source's then-current locked-in default, back
# when lam_mar/lam_da were tuned per source. configs/hparams.py now locks
# lam_mar=5, lam_da=3 UNIFORMLY across all 4 CMAPSS sources, so that design
# no longer brackets the actual deployed point (all its points were <=2.0/
# <=1.5, well below 5/3) -- redesigned to actually cover it.
#
# New range: (0.5, 0.1) to (5.0, 5.0), walking a path THROUGH the real
# production value rather than around it:
#   (0.5, 0.1)  -- low corner
#   (1.5, 1.0)  -- low-mid
#   (3.0, 2.0)  -- mid, approaching the deployed point
#   (5.0, 3.0)  -- THE ACTUAL LOCKED PRODUCTION VALUE (configs/hparams.py,
#                  all 4 CMAPSS sources) -- the point the old design missed
#   (5.0, 5.0)  -- high corner, both swept maxima
#
# Writes to a DEDICATED --logs_root (./logs_lambda_sensitivity), never
# ./logs or ./logs_abl, so this can never collide with the real campaign or
# ablation data.
#
# Usage:
#   bash scripts/lambda_sensitivity_cmapss.sh                # all 12 pairs
#   bash scripts/lambda_sensitivity_cmapss.sh FD001           # just FD001's 3 targets
#   bash scripts/lambda_sensitivity_cmapss.sh FD001 FD002     # just this one pair
#
# Save output:
#   nohup bash scripts/lambda_sensitivity_cmapss.sh > lambda_sensitivity.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# On this machine, `command -v python3` resolves to a broken Windows Store
# alias stub (exits 0 but prints "Python was not found..." and does
# nothing), so it must NOT be tried first the way da_gs_01.sh's pattern does.
PY=$(command -v python || command -v python3)
SCRIPT="CMAPSS_DA.py"
LOGS_ROOT="./logs_lambda_sensitivity"
SEED=42

ALL_SOURCES=("FD001" "FD002" "FD003" "FD004")
CONFIGS=("0.5,0.1" "1.5,1.0" "3.0,2.0" "5.0,3.0" "5.0,5.0")

# -- Parse optional CLI filters: 1 arg = source only (all its targets), 2 args = exact pair
FILTER_SRC="${1:-}"
FILTER_TGT="${2:-}"

PAIRS=()
for src in "${ALL_SOURCES[@]}"; do
    [[ -n "$FILTER_SRC" && "$src" != "$FILTER_SRC" ]] && continue
    for tgt in "${ALL_SOURCES[@]}"; do
        [[ "$tgt" == "$src" ]] && continue
        [[ -n "$FILTER_TGT" && "$tgt" != "$FILTER_TGT" ]] && continue
        PAIRS+=("${src}_${tgt}")
    done
done

total=$(( ${#PAIRS[@]} * ${#CONFIGS[@]} ))
done_count=0

echo "=================================================================="
echo "  lam_mar/lam_da sensitivity  |  LBGN_RUL  |  CMAPSS  |  $total experiments"
echo "  Pairs   : ${PAIRS[*]}"
echo "  Configs : ${CONFIGS[*]}  (lam_mar,lam_da)"
echo "  Seed    : $SEED"
echo "  Logs    : $LOGS_ROOT"
echo "=================================================================="

for pair in "${PAIRS[@]}"; do
    src="${pair%%_*}"
    tgt="${pair##*_}"
    for cfg in "${CONFIGS[@]}"; do
        lmar="${cfg%%,*}"
        lda="${cfg##*,}"
        done_count=$(( done_count + 1 ))
        tag="lmar${lmar}_lda${lda}_s${SEED}"
        ckpt="${LOGS_ROOT}/${pair}/LBGN_RUL/${tag}/best_checkpoint.pth"

        if [[ -f "$ckpt" ]]; then
            echo "SKIP [$done_count/$total] $pair | lam_mar=$lmar lam_da=$lda (done)"
            continue
        fi

        echo ""
        echo "RUN  [$done_count/$total] $pair | lam_mar=$lmar lam_da=$lda"
        $PY "$SCRIPT" \
            --model_name          LBGN_RUL \
            --dataset_name        CMAPSS \
            --Data_id_CMAPSS      "$src" \
            --Data_id_CMAPSS_test "$tgt" \
            --loss_type           DA \
            --seed                "$SEED" \
            --lam_mar             "$lmar" \
            --lam_da              "$lda" \
            --logs_root           "$LOGS_ROOT" \
            --save_path            "$tag" \
            --info                "LBGN_RUL_${pair}_${tag}_sensitivity" \
            || echo "FAIL [$done_count/$total] $pair | lam_mar=$lmar lam_da=$lda"
    done
done

echo ""
echo "=================================================================="
echo "  Done.  $done_count / $total experiments.  Results in: $LOGS_ROOT/"
echo "=================================================================="
