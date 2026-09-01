# LBGN-RUL: Lifecycle Band Graph Network for Cross-Condition Remaining Useful Life Prediction

> **Status: code and pretrained weights coming soon.** This repository is being prepared to accompany our manuscript, currently under submission. The implementation, trained checkpoints, and full experiment configs will be uploaded here shortly — watch/star the repo to be notified.

## Overview

LBGN-RUL is a graph neural network for **remaining useful life (RUL) prediction** of aero-engines under **cross-condition domain shift** (e.g., transferring a predictor trained on one C-MAPSS/N-CMAPSS operating subset to another). Instead of aligning a global feature embedding or the predicted output — the standard recipe in domain-adaptation approaches to RUL — LBGN-RUL makes the **inter-sensor coupling structure itself lifecycle-aware and band-resolved**, and aligns that structure across domains.

Key ideas:

- **Frequency-domain graph construction.** Inter-sensor coupling is decomposed into per-band coherence matrices (via the real FFT) rather than a single full-band, time-domain similarity score, so degradation-related structure that is concentrated in specific frequency bands is not averaged away.
- **Lifecycle conditioning at three points.** A lifecycle embedding, derived from the operational cycle count (OCC), drives (i) a soft weighting over the coherence bands, (ii) a depth-wise residual FiLM modulation of the time-view features at every propagation hop, and (iii) an independent FiLM correction inside a complementary spectral view that retains the aggregate coherence the band weighting discards.
- **Lifecycle-matched structural alignment.** A staged maximum mean discrepancy (MMD) penalty aligns the source/target adjacency stage-by-stage, using observable OCC quantiles rather than pseudo-labels, together with a monotonic coupling prior.

## Results summary

Evaluated against eight representative domain-adaptation baselines (DAGCN, DAST, MDAN, CADA, TACDA, EviAdapt, OCS-DANN, CCDG) across all cross-condition transfers on two aero-engine degradation benchmarks:

| Benchmark | Mean RMSE reduction vs. strongest baseline | Mean Score reduction vs. strongest baseline |
|---|---|---|
| C-MAPSS (12 transfers) | 28.5% | 55.3% |
| N-CMAPSS (12 transfers) | 49.1% | 75.0% |

LBGN-RUL also uses the fewest trainable parameters and the lowest FLOPs among all nine compared methods. Full per-transfer results, ablations, and statistical significance tests are reported in the paper.

## Repository contents (planned)

Once uploaded, this repository will include:

- `models/` — LBGN-RUL architecture and the reimplemented baselines used for comparison
- `experiments/` — training and evaluation scripts, including the ablation configurations (A1–A4) reported in the paper
- `data/` — preprocessing scripts for C-MAPSS and N-CMAPSS
- `configs/` — hyperparameter configurations for each dataset/source domain
- pretrained checkpoints for LBGN-RUL on all reported transfers

## Datasets

- **C-MAPSS** — NASA Prognostics Data Repository
- **N-CMAPSS** — NASA-affiliated PCoE data repository

Both are publicly available; links will be added alongside the code.

## Citation

A citation entry (BibTeX) will be added here once the paper is accepted/published. If you use this work in the meantime, please check back for the correct reference, or cite the manuscript title and authors below.

```
Lifecycle Band Graph Network for Cross-Condition Remaining Useful Life Prediction
Zeeshan Abbas, Hao Wang, Mehboob Hussain, Abid Hussain, Wenming Cao
(Manuscript under submission)
```

## License

License to be added upon code release.

## Contact

For questions in the meantime, please open an issue or contact the corresponding author (see the manuscript for details).
