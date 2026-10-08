# DA Baselines Reference — LBGN TII Paper

Source of truth for the comparison methods. Details extracted from the
implementation docstrings in `Models_RUL/` (which record the original repos
and adaptation choices). Verify citation years/venues against the original
papers before final submission.

## The 8 DA baselines

| Model | Full name / reference | Alignment type | What gets aligned | Lifecycle-aware? |
|---|---|---|---|---|
| DAGCN | Domain Adversarial Graph Convolutional Network (Li et al., fault diagnosis under variable working conditions; adapted to RUL) | adversarial (GRL) | global features (sample-graph ChebConv) | no |
| EviAdapt | Evidential Domain Adaptation for RUL with Incomplete Degradation (Li et al., IEEE TIM 2024) | evidential / discrepancy | NIG evidential feature distributions | no |
| DAST | Domain Adaptive RUL Prediction with Transformer (IEEE TIM 2022, doi 10.1109/TIM.2022.3200667) | adversarial (dual GRL discriminators) | global features (D2) + output RUL sequence (D1) | no |
| CADA | Contrastive Adversarial Domain Adaptation (mohamedr002/CADA, IEEE TIM) | ADDA + InfoNCE | global features (target encoder vs frozen source) | no |
| TACDA | Temporal Autoencoder Contrastive DA (keyplay/TACDA) | ADDA + temporal reconstruction | global features + reconstruction consistency | no |
| OCS-DANN | DA via Alignment of Operation Profile (Nejjar et al., 2023) | adversarial (DANN), per-sample soft-weighted by predicted operating condition | **condition-conditional** features | partially — OC proxied by 3 fixed OCC stage bins (0.15/0.66 cuts) |
| MDAN | Mixup Domain Adaptation Network (Furqon et al.) | mixup intermediate domain, adaptive λ via Wasserstein | feature-space mixture path | no |
| CCDG | Conditional Contrastive Domain Generalization (Mohamed et al., IEEE TIM 2022) | stage-conditional contrastive (τ=0.7) | **stage-conditional** features; target stages from RUL pseudo-labels | partially — same 3 fixed stage bins; target staging is pseudo-labelled |

## Implementation/protocol notes (for Sec. V and reproducibility)

- All baselines adapted into one framework (`Experiment.py`, per-baseline
  training loops matching original repos: `training_adversarial`, `training_dast`,
  `training_cada`, `training_tacda`, `training_ocs_dann`, `training_mdan`,
  `training_ccdg`, `training_eviadapt`). Hyperparameters from original
  papers/repos; LR grid-searched per source (see `configs/hparams.py` comments).
- Notable faithful-adaptation deviations to disclose if asked:
  DAGCN ChebConv re-implemented dense (no torch_geometric), sample-graph as in
  original; TACDA uses MSE reconstruction in place of SoftDTW (dependency);
  DAST original d_model=24→ours 32 with input projection 14→32.
- OCS-DANN original defines OC as N-CMAPSS flight phases; CMAPSS adaptation
  proxies OC with OCC-derived lifecycle stages (early ≤0.15 < mid ≤0.66 < late).

## Non-DA lifecycle-aware references (ancestry, not competitors)

- **AIDGN**: OCC-gated adjacency; no DA mechanism.
- **DLGNet**: OCC-conditioned attention adjacency (ACE) + graph-Laplacian
  smoothing (SASRLayer, node-feature space — NOT frequency-domain); no FiLM,
  no DA mechanism.

## Gap statement implications (IMPORTANT for the paper)

Two baselines are NOT fully lifecycle-blind: OCS-DANN (stage-soft-weighted
adversarial loss) and CCDG (stage-conditional contrastive). The gap must
therefore be phrased precisely:

1. ALL eight baselines align FEATURE (or output) distributions — none aligns
   the inter-sensor coupling STRUCTURE a graph predictor relies on. This is
   the clean universal differentiator for C3.
2. The stage-aware ones (OCS-DANN, CCDG) use three FIXED, hand-cut OCC bins
   as a conditioning label for feature alignment; CCDG's target stages are
   pseudo-labelled from the model's own predictions. LBGN's staging is
   quantile-based on observable OCC in both domains (no pseudo-labels), and
   lifecycle additionally conditions the REPRESENTATION (adjacency + FiLM),
   not just the alignment loss.
3. Never write "existing DA methods ignore lifecycle stage" — OCS-DANN and
   CCDG partially don't, and they are in our own table. Write: "either ignore
   lifecycle position or use it only as a coarse conditioning label for
   feature-level alignment."
