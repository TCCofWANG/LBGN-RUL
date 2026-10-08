# Analysis scripts

Everything in this folder is a read-only consumer of trained checkpoints
(`logs/**/result.npz`, `logs/**/best_checkpoint.pth`) and grid-search CSVs
(`*experimental_logs.csv`). Nothing in the main codebase imports from here
— deleting this folder removes the analysis code with zero side effects.

## Main results table

| Script | Output | Reads from |
|---|---|---|
| `build_local_final_table.py` | `out/local_final_table_{rmse,score,seed_counts}.csv` | `logs/{pair}/{model}/final_s{seed}/result.npz` directly (bypasses the flat CSVs, which are missing the final campaign for some models — see the script's own docstring) |

## Ablation tables

| Script | Output | Dataset |
|---|---|---|
| `build_ablation_results_table.py` + `build_ablation_latex_table.py` | `out/DA_RUL_ablation_table.tex`, `out/DA_RUL_ablation_results_transposed.xlsx` | CMAPSS |
| `build_ablation_results_table_ncmapss.py` + `build_ablation_latex_table_ncmapss.py` | `out/DA_RUL_NCMAPSS_ablation_table.tex`, `out/DA_RUL_NCMAPSS_ablation_results_transposed.xlsx` | N-CMAPSS |
| `ablation_ndc_r1_from_ckpts.py` | printed report | NDC_PDMN r=1 ablation, from checkpoints (companion note: `scripts/run_ablation_ndc_r1.sh` produces the ablation runs first) |

## Lineage comparison

| Script | Output |
|---|---|
| `build_lineage_comparison_latex.py` | `out/DA_RUL_lineage_comparison_table.tex` — LBGN-RUL (Full) vs. its direct architectural predecessors AIDGN and DLGNet |

## Figures

| Script | Figure | Inputs |
|---|---|---|
| `fig_film_gain_vs_occ.py` | Per-depth FiLM gain vs lifecycle position | one trained checkpoint |
| `fig_rmse_vs_capacity.py` | Best RMSE vs hidden_dim per model | folder of `*experimental_logs.csv` grid logs |
| `fig_adjacency_domain_stability.py` | Adjacency stability across domains | trained checkpoints |
| `fig_lambda_sensitivity.py` / `fig_lambda_sensitivity_per_pair.py` | RMSE/Score vs. loss-weight (lambda) sweeps | `logs_lambda_sensitivity/` |
| `fig_error_distribution.py` | Best/worst-case error distributions | `result.npz` |
| `plot_efficiency_bubble.py` | RMSE vs. inference time/FLOPs bubble chart | `out/flops_per_model.csv`, `out/inference_time_per_model.csv` |
| `plot_raw_cmapss_sensors.py` | Raw sensor traces | `CMAPSS/train_FD00X.txt` |
| `build_ndc_inference_dump.py` | Per-sample inference dump (engine_id, OCC, tau, ...) for NDC_PDMN figures | one trained checkpoint + raw `CMAPSS/` data |

## Other

| Script | Purpose |
|---|---|
| `build_transposed_ncmapss_gridsearch.py` | N-CMAPSS hyperparameter grid-search results table (distinct from the main results table above) |
| `flop_utils.py` | FLOPs/MACs counting utility, used by `scripts/measure_flops_cmapss.py` |

## Usage examples

```bash
python analysis/build_local_final_table.py --logs_dir logs --out analysis/out/local_final_table

python analysis/fig_film_gain_vs_occ.py \
    --ckpt logs/FD002_FD004/FiLMHeavyTimeView_RUL/final_s0/best_checkpoint.pth \
    --lce_dim 8 --out analysis/out

python analysis/fig_rmse_vs_capacity.py \
    --logs_dir "./external_data" --out analysis/out
```
