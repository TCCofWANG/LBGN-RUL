#!/usr/bin/env bash
# =============================================================================
# N-CMAPSS check of the no_da loss-level ablation (lambda_mar=0 AND
# lambda_da=0 together) -- LBGN_RUL only. Follows up on the CMAPSS 12-pair
# no_da sweep (2026-07-28/29, see abl_logs).
#
# Only runs 'no_da'. No 'none' reference here -- --lbgn_ablation none is
# just the default loss config (whatever configs/hparams.py sets for this
# model/dataset), so run_ncmapss_da.sh's existing LBGN_RUL final_s42 row
# already IS the none baseline. That row is a genuine cold start (its
# self-referential --warm_start_checkpoint only ever gets reached AFTER the
# checkpoint exists, since the resume-skip check runs first), so it's valid
# to diff against without rerunning it here.
#
# COLD START (no --resume/--warm_start_checkpoint), in its own dedicated
# logs_root (./abl_logs_NCMAPSS, never ./logs_NCMAPSS) so it can't collide
# with or be mistaken for a main-table row.
#
# IMPORTANT: only seed 42 has a clean cold-start none baseline to compare
# against (final_s42). run_ncmapss_da.sh's OTHER seeds are warm-started from
# final_s42, not cold, so a no_da run at any other seed here has no
# apples-to-apples none counterpart yet -- stick to seed 42 unless you also
# cold-start a matching none reference for whatever other seeds you add.
#
# configs/hparams.py's N_CMAPSS entries are still UNTESTED placeholders
# (copied unchanged from CMAPSS FD001, num_nodes 14->20) -- treat this as a
# first look, not a tuned result, same caveat as run_ncmapss_da.sh.
#
# Usage:
#   bash scripts/NCMAPSS/run_no_da_ncmapss.sh <SRC> [seeds...]
#   bash scripts/NCMAPSS/run_no_da_ncmapss.sh DS01          # seed 42 only (default)
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SRC="${1:?usage: run_no_da_ncmapss.sh SRC [seeds...]}"
shift || true

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
MODEL="LBGN_RUL"
LOGS_ROOT="./abl_logs_NCMAPSS"

VARIANTS=("no_da")

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(42)
fi

ALL_DS=("DS01" "DS02" "DS03" "DS04")
TARGETS=()
for ds in "${ALL_DS[@]}"; do
    [[ "$ds" == "$SRC" ]] && continue
    TARGETS+=("$ds")
done

total=$(( ${#VARIANTS[@]} * ${#TARGETS[@]} * ${#SEEDS[@]} ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  N-CMAPSS no_da check  |  SRC: $SRC  |  $total experiments"
echo "  Targets : ${TARGETS[*]}"
echo "  Seeds   : ${SEEDS[*]}"
echo "  Logs -> $LOGS_ROOT/  (kept separate from logs_NCMAPSS -- never mixed)"
echo "  Compare against: logs_NCMAPSS/{SRC}_{TGT}/LBGN_RUL/final_s42/ (the none baseline)"
echo "══════════════════════════════════════════════════════════════"

for TGT     in "${TARGETS[@]}";  do
for variant in "${VARIANTS[@]}"; do
for SEED    in "${SEEDS[@]}";    do

    done_count=$(( done_count + 1 ))
    tag="abl_${variant}_s${SEED}"
    ckpt="${LOGS_ROOT}/${SRC}_${TGT}/${MODEL}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $TGT | $variant | seed $SEED (done)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $TGT | $variant | seed $SEED"

    $PY "$SCRIPT" \
        --model_name             "$MODEL"    \
        --dataset_name           N_CMAPSS    \
        --Data_id_N_CMAPSS       "$SRC"      \
        --Data_id_N_CMAPSS_test  "$TGT"      \
        --seed                   "$SEED"     \
        --save_path              "$tag"      \
        --logs_root              "$LOGS_ROOT" \
        --lbgn_ablation          "$variant"  \
        --info                   "${MODEL}_${SRC}_${TGT}_seed${SEED}_ablation_${variant}" \
        || echo "✗  [FAIL] $SRC → $TGT | $variant | seed $SEED"

done; done; done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  SRC $SRC no_da check complete.  Results in: $LOGS_ROOT/"
echo "══════════════════════════════════════════════════════════════"
