#!/usr/bin/env bash
# =============================================================================
# N-CMAPSS check of the no_spectral_diffusion RQ5 ablation (spectral view's
# one-step Laplacian diffusion pass skipped, z_out = z) -- LBGN_RUL only.
# Follows up on the CMAPSS 12-pair no_spectral_diffusion sweep (logs_abl/,
# 8-seed campaign: -0.19 RMSE, p=0.29 -- not significant, consistent with the
# paper's claim that the diffusion step itself is borrowed/non-novel
# machinery, not part of contribution 2). See Models_RUL/LBGN_RUL_ablation.py
# for the variant's implementation (SpectralGraphLayerNoDiffusion).
#
# Only runs 'no_spectral_diffusion'. No 'none' reference here -- same
# rationale as run_no_da_ncmapss.sh: run_ncmapss_da.sh's existing LBGN_RUL
# final_s42 row already IS the none baseline (genuine cold start), so it's
# valid to diff against without rerunning it here.
#
# Seed protocol (matches the CMAPSS local campaign's run_ablation_FD00X.py,
# NOT run_no_da_ncmapss.sh's flat independent-cold-starts): COLD_SEED (42) is
# a genuine cold start (--resume False); every WARM_SEED is warm-started FROM
# that same variant's own seed-42 checkpoint (--resume True --resume_path
# abl_no_spectral_diffusion_s42), per target. This is safe within this one
# variant (identical architecture across its own seeds); never done across
# variants. If seed 42 fails for a target, that target's warm-start seeds are
# skipped rather than attempted (would otherwise crash on a missing
# checkpoint).
#
# COLD START logs_root is ./abl_logs_NCMAPSS (never ./logs_NCMAPSS) so it
# can't collide with or be mistaken for a main-table row. Shares the logs
# root with the no_da check (run_no_da_ncmapss.sh) -- no collision, since the
# variant name is baked into each run's tag/checkpoint path.
#
# Resume-safe: skips a (target, seed) combo if its
# abl_no_spectral_diffusion_s{seed}/best_checkpoint.pth already exists.
#
# configs/hparams.py's N_CMAPSS entries are still UNTESTED placeholders
# (copied unchanged from CMAPSS FD001, num_nodes 14->20) -- treat this as a
# first look, not a tuned result, same caveat as run_ncmapss_da.sh.
#
# Usage:
#   bash scripts/NCMAPSS/run_no_spectral_diffusion_ncmapss.sh <SRC> [warm_seeds...]
#   bash scripts/NCMAPSS/run_no_spectral_diffusion_ncmapss.sh DS01                       # seed 42 only (cold)
#   bash scripts/NCMAPSS/run_no_spectral_diffusion_ncmapss.sh DS01 0 1 7 41 64 1234 2026  # full 8-seed set
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SRC="${1:?usage: run_no_spectral_diffusion_ncmapss.sh SRC [warm_seeds...]}"
shift || true

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
MODEL="LBGN_RUL"
LOGS_ROOT="./abl_logs_NCMAPSS"
VARIANT="no_spectral_diffusion"
COLD_SEED=42
WARM_SEEDS=("$@")   # empty = cold-start seed 42 only, same as before

ALL_DS=("DS01" "DS02" "DS03" "DS04")
TARGETS=()
for ds in "${ALL_DS[@]}"; do
    [[ "$ds" == "$SRC" ]] && continue
    TARGETS+=("$ds")
done

total=$(( ${#TARGETS[@]} * (1 + ${#WARM_SEEDS[@]}) ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  N-CMAPSS no_spectral_diffusion check  |  SRC: $SRC  |  $total experiments"
echo "  Targets    : ${TARGETS[*]}"
echo "  Cold seed  : $COLD_SEED"
echo "  Warm seeds : ${WARM_SEEDS[*]:-(none -- cold seed 42 only)}"
echo "  Logs -> $LOGS_ROOT/  (kept separate from logs_NCMAPSS -- never mixed)"
echo "  Compare against: logs_NCMAPSS/{SRC}_{TGT}/LBGN_RUL/final_s42/ (the none baseline)"
echo "══════════════════════════════════════════════════════════════"

run_one () {
    local TGT="$1" SEED="$2" RESUME="$3" RESUME_PATH="$4" TAG="$5"
    $PY "$SCRIPT" \
        --model_name             "$MODEL"      \
        --dataset_name           N_CMAPSS      \
        --Data_id_N_CMAPSS       "$SRC"        \
        --Data_id_N_CMAPSS_test  "$TGT"        \
        --seed                   "$SEED"       \
        --save_path              "$TAG"        \
        --logs_root              "$LOGS_ROOT"  \
        --resume                 "$RESUME"     \
        --resume_path            "$RESUME_PATH" \
        --lbgn_ablation          "$VARIANT"    \
        --info                   "${MODEL}_${SRC}_${TGT}_seed${SEED}_ablation_${VARIANT}"
}

for TGT in "${TARGETS[@]}"; do

    cold_tag="abl_${VARIANT}_s${COLD_SEED}"
    cold_ckpt="${LOGS_ROOT}/${SRC}_${TGT}/${MODEL}/${cold_tag}/best_checkpoint.pth"

    done_count=$(( done_count + 1 ))
    if [[ -f "$cold_ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $TGT | $VARIANT | seed $COLD_SEED (done)"
    else
        echo ""
        echo "▶  [$done_count/$total] $SRC → $TGT | $VARIANT | seed $COLD_SEED (cold start)"
        run_one "$TGT" "$COLD_SEED" False "$cold_tag" "$cold_tag" \
            || echo "✗  [FAIL] $SRC → $TGT | $VARIANT | seed $COLD_SEED"
    fi

    if [[ ! -f "$cold_ckpt" ]]; then
        echo "⚠  seed $COLD_SEED checkpoint still missing for $SRC → $TGT -- skipping its ${#WARM_SEEDS[@]} warm-start seeds"
        done_count=$(( done_count + ${#WARM_SEEDS[@]} ))
        continue
    fi

    for SEED in "${WARM_SEEDS[@]}"; do
        tag="abl_${VARIANT}_s${SEED}"
        ckpt="${LOGS_ROOT}/${SRC}_${TGT}/${MODEL}/${tag}/best_checkpoint.pth"

        done_count=$(( done_count + 1 ))
        if [[ -f "$ckpt" ]]; then
            echo "⏭  [$done_count/$total] skip $SRC → $TGT | $VARIANT | seed $SEED (done)"
            continue
        fi

        echo ""
        echo "▶  [$done_count/$total] $SRC → $TGT | $VARIANT | seed $SEED (warm start from seed $COLD_SEED)"
        run_one "$TGT" "$SEED" True "$cold_tag" "$tag" \
            || echo "✗  [FAIL] $SRC → $TGT | $VARIANT | seed $SEED"
    done

done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  SRC $SRC no_spectral_diffusion check complete.  Results in: $LOGS_ROOT/"
echo "══════════════════════════════════════════════════════════════"
