#!/usr/bin/env bash
# =============================================================================
# N-CMAPSS LR x seed grid search -- DS01-04 only for now (widen ALL_SOURCES
# to include DS05-07 later if a reviewer asks for the rest).
#
# configs/hparams.py's N_CMAPSS train_params are currently UNTESTED placeholder
# values copied unchanged from CMAPSS's FD001 block (num_nodes 14->20) -- this
# sweep exists to find a per-source LR that's actually stable on N-CMAPSS
# before locking it in, the same way scripts/grid_search_seeds_FD001.sh did
# for CMAPSS.
#
# IMPORTANT: this passes --learning_rate_override, NOT --learning_rate.
# CMAPSS_DA.py (the shared entry point for CMAPSS and N-CMAPSS alike --
# dispatch is via --dataset_name) applies configs/hparams.py's train_params
# AFTER argparse, which unconditionally overwrites args.learning_rate with
# the model's tuned/placeholder value -- so a plain --learning_rate here would
# silently be clobbered and every run would use the same hparams.py value
# regardless of $lr, making the whole sweep a no-op. --learning_rate_override
# is applied after that merge specifically to survive it.
#
# LBGN_RUL also gets a hidden_dim=64 variant alongside the default 32 --
# it's a much lighter model than the baselines, and N-CMAPSS has more sensor
# channels (num_nodes 20 vs CMAPSS's 14) than it was tuned against, so it's
# worth checking whether the extra capacity helps here specifically. No other
# model gets this extra sweep (same pattern grid_search_seeds_FD001.sh used
# for NDC_PDMN's hidden_dim check on CMAPSS).
#
# Cold-start, resume-safe (skips a (pair, model, lr, seed[, hd]) combo if its
# checkpoint already exists). Writes to a DEDICATED logs_root (./ncmapss_lr_logs),
# never ./logs/, so it can never collide with the real campaign. info is NOT
# tagged "_final" -- scripts/copy_best_checkpoints.py already treats untagged
# rows as one flat candidate pool per (pair, model), which is exactly what a
# grid search is.
#
# Single seed (0) here, not a seed sweep -- N-CMAPSS is too expensive per run
# to afford a stability-across-seeds check at this stage (that's what
# scripts/grid_search_seeds_FD001.sh did for CMAPSS: sweep LR x multiple
# seeds specifically to find an LR robust to catastrophic seed blow-ups).
# Trade-off: an LR that looks good at seed=0 could still blow up on a
# different seed once scripts/NCMAPSS/warmstart_from_best.sh runs its wider
# seed set -- if that happens for a given source, come back and rerun this
# script for just that source with SEEDS widened (e.g. SEEDS=(0 42)) before
# trusting its locked-in LR.
#
# N-CMAPSS is much larger per-sample than CMAPSS -- the grid below
# (~432 runs: 10 non-LBGN models x 4 sources x 3 targets x 3 LR x 1 seed,
# plus LBGN_RUL x ... x 2 hidden_dims) is still a lot. Scope it down further
# with CLI filters before committing, e.g.:
#   bash scripts/NCMAPSS/grid_search_lr_seed.sh LBGN_RUL DS02   # one model, one source
#   bash scripts/NCMAPSS/grid_search_lr_seed.sh DS02 DS03       # two sources, all models
#
# Save output:
#   nohup bash scripts/NCMAPSS/grid_search_lr_seed.sh > ncmapss_gs.log 2>&1 &
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=$(command -v python3 || command -v python)
SCRIPT="CMAPSS_DA.py"
LOGS_ROOT="./ncmapss_lr_logs"

ALL_MODELS=("LBGN_RUL" "PDMN" "NDC_PDMN" "DAGCN_RUL" "EviAdaptRUL" "DAST_RUL" "CADA_RUL" "TACDA_RUL" "OCS_DANN" "MDAN_RUL" "CCDG_RUL" "AIDGN" "DLGNet")
ALL_SOURCES=("DS01" "DS02" "DS03" "DS04")

LR_VALUES=(0.005 0.001 0.0005)
SEEDS=(42)
LBGN_HIDDEN_DIMS=(32 64)

# -- Parse optional CLI filters (model names vs DS-prefixed source filters)
MODELS=()
SOURCES=()
for arg in "$@"; do
    [[ "$arg" == DS* ]] && SOURCES+=("$arg") || MODELS+=("$arg")
done
[[ ${#MODELS[@]}  -eq 0 ]] && MODELS=("${ALL_MODELS[@]}")
[[ ${#SOURCES[@]} -eq 0 ]] && SOURCES=("${ALL_SOURCES[@]}")

has_model() { [[ " ${MODELS[*]} " == *" $1 "* ]]; }

# per-(src,tgt)-pair run count: every model in MODELS gets LR x SEEDS runs;
# LBGN_RUL additionally multiplies by LBGN_HIDDEN_DIMS instead of running once.
per_pair=0
for m in "${MODELS[@]}"; do
    if [[ "$m" == "LBGN_RUL" ]]; then
        per_pair=$(( per_pair + ${#LR_VALUES[@]} * ${#SEEDS[@]} * ${#LBGN_HIDDEN_DIMS[@]} ))
    else
        per_pair=$(( per_pair + ${#LR_VALUES[@]} * ${#SEEDS[@]} ))
    fi
done

n_pairs=0
for src in "${SOURCES[@]}"; do
    for tgt in "${ALL_SOURCES[@]}"; do
        [[ "$tgt" == "$src" ]] && continue
        n_pairs=$(( n_pairs + 1 ))
    done
done
total=$(( per_pair * n_pairs ))
done_count=0

echo "══════════════════════════════════════════════════════════════"
echo "  N-CMAPSS LR x seed grid search  |  $total experiments"
echo "  Models       : ${MODELS[*]}"
echo "  Sources      : ${SOURCES[*]}  (targets = other DS01-04 members)"
echo "  LR           : ${LR_VALUES[*]}"
echo "  Seeds        : ${SEEDS[*]}"
echo "  LBGN hidden_dim : ${LBGN_HIDDEN_DIMS[*]}  (LBGN_RUL only)"
echo "  Logs -> $LOGS_ROOT/"
echo "══════════════════════════════════════════════════════════════"

run_exp() {
    local model="$1" src="$2" tgt="$3" lr="$4" seed="$5" tag="$6"
    shift 6
    done_count=$(( done_count + 1 ))
    local ckpt="${LOGS_ROOT}/${src}_${tgt}/${model}/${tag}/best_checkpoint.pth"

    if [[ -f "$ckpt" ]]; then
        echo "⏭  [$done_count/$total] skip $src → $tgt | $model | $tag (done)"
        return
    fi

    echo ""
    echo "▶  [$done_count/$total] $src → $tgt | $model | $tag"

    $PY "$SCRIPT" \
        --model_name             "$model"     \
        --dataset_name           N_CMAPSS     \
        --Data_id_N_CMAPSS       "$src"       \
        --Data_id_N_CMAPSS_test  "$tgt"       \
        --learning_rate_override "$lr"        \
        --seed                   "$seed"      \
        --logs_root              "$LOGS_ROOT" \
        --save_path               "$tag"      \
        --info                   "${model}_${src}_${tgt}_${tag}_gridsearch" \
        "$@" \
        || echo "✗  [FAIL] $src → $tgt | $model | $tag"
}

for src in "${SOURCES[@]}"; do
for tgt in "${ALL_SOURCES[@]}"; do
    [[ "$tgt" == "$src" ]] && continue
for model in "${MODELS[@]}"; do
for lr in "${LR_VALUES[@]}"; do
for seed in "${SEEDS[@]}"; do

    if [[ "$model" == "LBGN_RUL" ]]; then
        for hd in "${LBGN_HIDDEN_DIMS[@]}"; do
            run_exp "$model" "$src" "$tgt" "$lr" "$seed" "gs_hd${hd}_lr${lr}_seed${seed}" --hidden_dim_override "$hd"
        done
    else
        run_exp "$model" "$src" "$tgt" "$lr" "$seed" "gs_lr${lr}_seed${seed}"
    fi

done; done; done; done; done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Grid search complete.  Results in: $LOGS_ROOT/"
echo "  Next steps:"
echo "    1. Inspect $LOGS_ROOT/*experimental_logs.csv, pick the LR that's"
echo "       stable per source (and for LBGN_RUL, the better of hd32/hd64),"
echo "       lock it into configs/hparams.py's N_CMAPSS train_params/alg_hparams"
echo "       entry for that source."
echo "    2. python scripts/copy_best_checkpoints.py --logs_dir $LOGS_ROOT \\"
echo "           --dataset_name N_CMAPSS --out ./ncmapss_logs_best"
echo "    3. bash scripts/NCMAPSS/warmstart_from_best.sh"
echo "══════════════════════════════════════════════════════════════"
