#!/usr/bin/env bash
# =============================================================================
# Regenerate result.npz for already-trained NDC_PDMN checkpoints (full model
# + r=1 ablation) by loading the checkpoint and running a real inference pass
# -- no retraining. Needed wherever result.npz is missing or you want fresh
# predictions to feed analysis/bootstrap_ablation_test.py.
#
# How this works: CMAPSS_DA.py's Experiment.start() calls self.test(...)
# UNCONDITIONALLY (outside the `if self.args.train:` block), and test() saves
# result.npz whenever --save_test is on (default True). CSV-row writing IS
# gated by `if self.args.train:`, so --train False --resume True never adds
# a duplicate row to experimental_logs.csv -- this is a pure read+eval, safe
# to run any number of times.
#
# Resume-safe: skips a (pair, tag) combo if result.npz already exists,
# unless --force is passed as the first argument.
#
# Usage:
#   bash scripts/regen_result_npz.sh            # only fill in missing npz
#   bash scripts/regen_result_npz.sh --force     # regenerate everything
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
MODEL="NDC_PDMN"
SEEDS=(0 42 64 7 123)

PAIRS=(
    "FD001 FD002" "FD001 FD003" "FD001 FD004"
    "FD002 FD001" "FD002 FD003" "FD002 FD004"
    "FD003 FD001" "FD003 FD002" "FD003 FD004"
    "FD004 FD001" "FD004 FD002" "FD004 FD003"
)

total=$(( ${#PAIRS[@]} * ${#SEEDS[@]} * 2 ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  Regenerating result.npz  |  force=$FORCE  |  $total (pair, tag) combos"
echo "══════════════════════════════════════════════════════════════"

for pair in "${PAIRS[@]}"; do
    read -r SRC tgt <<< "$pair"
for seed in "${SEEDS[@]}"; do
for kind in final ablation; do

    done_count=$(( done_count + 1 ))
    if [[ "$kind" == "final" ]]; then
        tag="final_s${seed}"
        extra_flag=""
    else
        tag="ablation_r1_s${seed}"
        extra_flag="--ndc_ablate_rate1"
    fi

    ckpt="./logs/${SRC}_${tgt}/${MODEL}/${tag}/best_checkpoint.pth"
    npz="./logs/${SRC}_${tgt}/${MODEL}/${tag}/result.npz"

    if [[ ! -f "$ckpt" ]]; then
        echo "⚠  [$done_count/$total] no checkpoint: $ckpt -- skipping"
        continue
    fi

    if [[ -f "$npz" && "$FORCE" == "false" ]]; then
        echo "⏭  [$done_count/$total] skip $SRC → $tgt | $tag (result.npz already exists)"
        continue
    fi

    echo ""
    echo "▶  [$done_count/$total] $SRC → $tgt | $tag"

    $PY "$SCRIPT" \
        --model_name          "$MODEL"   \
        --dataset_name        CMAPSS     \
        --Data_id_CMAPSS      "$SRC"     \
        --Data_id_CMAPSS_test "$tgt"     \
        --seed                "$seed"    \
        --train                False     \
        --resume                True     \
        --resume_path          "$tag"    \
        --save_path            "$tag"    \
        $extra_flag \
        || echo "✗  [FAIL] $SRC → $tgt | $tag"

done; done; done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Done. result.npz refreshed under ./logs/{pair}/NDC_PDMN/{tag}/"
echo "════════════════════════════════════════════════════════════════"
