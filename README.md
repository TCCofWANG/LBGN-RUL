# LBGN-RUL

Lifecycle Band Graph Network for Remaining Useful Life (RUL) prediction under
domain shift, evaluated on the CMAPSS and N-CMAPSS turbofan degradation
datasets.

## Overview

LBGN-RUL combines two views of the sensor signal:

- **Time view** — a lifecycle-band-coherence adjacency with depth-wise FiLM
  conditioning (`Models_RUL/LBGN_RUL.py`).
- **Spectral view** — an OCC-FiLM-conditioned spectral branch with a
  Laplacian relaxation step, fused with the time view via additive gating.

Domain adaptation is handled with a monotonic adjacency regularizer (MAR)
plus an adjacency-MMD structural-alignment term.

## Repository layout

- `CMAPSS_DA.py` — main entry point (training/evaluation for both datasets).
- `Experiment/Experiment.py` — training loop, data dispatch, model dispatch.
- `Models_RUL/` — LBGN-RUL and all compared baselines (AIDGN, DLGNet, PDMN,
  NDC_PDMN, DAGCN_RUL, EviAdaptRUL, DAST_RUL, CADA_RUL, TACDA_RUL, OCS_DANN,
  MDAN_RUL, CCDG_RUL, working_model_RUL).
- `Models_RUL/LBGN_RUL_ablation.py` — architectural ablation variants (A1
  adjacency substrate, A2 FiLM conditioning, A3 spectral view); loss-level
  ablations (A4: plain MSE, no-DA, source-only) are handled directly in
  `Experiment.training_da`.
- `CMAPSS_Related/`, `N_CMAPSS_Related/` — dataset loaders.
- `configs/` — per-dataset/per-model hyperparameters (`hparams.py`) and
  dataset shape config (`data_model_configs.py`).
- `scripts/` — run scripts for the full seed/pair sweep, ablations, and
  reproducibility checks (grid search, lambda sensitivity, FLOPs/inference
  timing).
- `analysis/` — scripts that turn raw experiment CSV logs into the paper's
  tables and figures.

## Setup

```bash
pip install -r requirements.txt
```

Point `--data_path_CMAPSS` at your local CMAPSS directory (default `./CMAPSS`).
N-CMAPSS paths are configured similarly via `--Data_id_N_CMAPSS` and the
loader in `N_CMAPSS_Related/`.

## Running

```bash
# Full model, single source -> target transfer
python CMAPSS_DA.py --model_name LBGN_RUL --dataset_name CMAPSS \
    --Data_id_CMAPSS FD001 --Data_id_CMAPSS_test FD003 --seed 42

# An architecture ablation
python CMAPSS_DA.py --model_name LBGN_RUL --lbgn_ablation no_lifecycle \
    --dataset_name CMAPSS --Data_id_CMAPSS FD001 --Data_id_CMAPSS_test FD003

# A baseline
python CMAPSS_DA.py --model_name AIDGN --dataset_name CMAPSS \
    --Data_id_CMAPSS FD001 --Data_id_CMAPSS_test FD003 --seed 42
```

See `scripts/` for the full 4-source x 3-target x 8-seed sweep used to
produce the paper's results, and `analysis/README.md` for how the result
CSVs are turned into tables.

## Reproducing the reported LBGN-RUL results

`release_artifacts/checkpoints/` ships the actual trained checkpoints behind
every LBGN-RUL number reported for CMAPSS — all 12 source→target pairs, all
8 seeds (96 checkpoints, ~6 MB total), each alongside the `hparam.yaml` it
was trained with and the `experimental_logs.csv` row it originally produced.
This is an exact, not approximate, reproduction path: reloading a checkpoint
re-runs the same forward pass on the same test set, so there is no
retraining-time source of variance (seed, hardware, driver version, cuDNN
algorithm selection) to reproduce around.

```bash
python test_CMAPSS_DA.py --logs_dir ./release_artifacts/checkpoints
```

This reloads all 96 checkpoints and re-evaluates each one, comparing the
freshly computed RMSE/Score against what's in the shipped
`experimental_logs.csv` rows. Verified: 96/96 reload and match to within
floating-point noise (no errors, no discrepancies) as of the last check.

Narrow it down with `--models LBGN_RUL --pairs FD001_FD002`, or point
`--logs_dir` at your own `logs/` tree after running the training commands
above to verify a fresh run the same way.

## Reproducibility check (general)

```bash
python test_CMAPSS_DA.py --logs_dir <any logs tree>
```

Reloads every checkpoint under `--logs_dir` and re-evaluates it, comparing
against the RMSE/Score originally recorded for that run. Nothing on disk is
modified — this is a read-only check.
