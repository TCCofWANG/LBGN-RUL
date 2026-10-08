"""
GitHub: OCS-DANN: Domain Adversarial Neural Network with Operation-Condition-Soft alignment.
paper: OPS-DANN:Domain adaptation via alignment of operation profile for Remaining Useful Lifetime prediction
Based on: Nejjar et al., "Domain Adaptation via Alignment of Operation Profile
for Remaining Useful Lifetime Prediction" (2023).
https://github.com/ismailnejjar/Domain-adaptation-via-alignment-of-operation-profile-for-remaining-useful-lifetime-prediction

Key idea vs plain DANN:
  The domain adversarial loss is per-sample soft-weighted by the model's own
  operating-condition (OC) predictions so that the discriminator aligns
  condition-specific feature distributions instead of the global distribution.

  Original paper defines OC as N-CMAPSS flight phases (ascending/steady/descending).
  For CMAPSS we proxy OC with three degradation lifecycle stages derived from OCC:
      stage 0 (early)  : OCC_norm ≤ 0.15
      stage 1 (mid)    : 0.15 < OCC_norm ≤ 0.66
      stage 2 (late)   : OCC_norm > 0.66

Architecture:
  Encoder   : 3 × Conv1d(C, k=5) + BN + ReLU → AdaptiveAvgPool → FC(hidden_dim)
  RUL head  : FC(hid → hid//2 → 1) + Sigmoid
  OC head   : FC(hid → 3)          [trained on source with OCC-derived pseudo-labels]
  Domain disc: GRL → FC(hid → 64 → 1)  [single shared discriminator, soft OC weighting]

Training (da_mode='ocs_dann'):
  loss = MSE(src) + lam_oc * CE(oc_src) + lam_da * (oc_soft_BCE_src + oc_soft_BCE_tgt)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _grl_hook(coeff: float):
    def _hook(grad):
        return -coeff * grad.clone()
    return _hook


def _grl_coeff(step, high=1.0, low=0.0, alpha=10.0, max_iter=10_000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * step / max_iter))
                 - (high - low) + low)


def occ_to_stage(occ_raw, max_life: float) -> torch.Tensor:
    """
    occ_raw : (B,) raw cycle index (scalar per sample, mean over window)
    Returns : (B,) LongTensor stage ∈ {0, 1, 2}
    """
    occ_n = occ_raw.float() / max_life
    s = torch.zeros(occ_n.shape[0], dtype=torch.long, device=occ_n.device)
    s[occ_n > 0.15] = 1
    s[occ_n > 0.66] = 2
    return s


class OCS_DANN(nn.Module):
    """
    CNN + OC-aware DANN for cross-domain RUL prediction.
    forward(x, OCC=None) → (None, {'gamma': pred, 'h': features})
    """
    def __init__(self, args):
        super().__init__()
        in_c       = getattr(args, 'num_nodes', 14)
        hid        = getattr(args, 'hidden_dim', 32)
        dropout    = getattr(args, 'dropout', 0.1)
        self.hidden_dim = hid
        self._grl_step  = 0


        self.encoder = nn.Sequential(
            nn.Conv1d(in_c, 32,  kernel_size=5, padding=2), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,   64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,   hid, kernel_size=5, padding=2), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(dropout))


        self.regressor = nn.Sequential(
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid // 2, 1),  nn.Sigmoid(),
        )


        self.oc_head = nn.Sequential(
            nn.Linear(hid, hid // 2), nn.ReLU(),
            nn.Linear(hid // 2, 3),
        )


        self.domain_disc = nn.Sequential(
            nn.Linear(hid, 64), nn.ReLU(),
            nn.Linear(64, 1),   nn.Sigmoid(),
        )


    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → h: (B, hidden_dim)"""
        h = self.encoder(x.permute(0, 2, 1))
        return self.proj(h)

    def _apply_grl(self, feat: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._grl_step += 1
        coeff = _grl_coeff(self._grl_step)
        feat_grl = feat * 1.0
        feat_grl.register_hook(_grl_hook(coeff))
        return feat_grl


    def forward(self, x, OCC=None):
        h    = self._encode(x)
        pred = self.regressor(h)
        return None, {'gamma': pred, 'h': h}

    def forward_da(self, x_s, x_t):
        """
        Returns source/target features, source OC logits, source/target domain logits.
        Called from training_ocs_dann in Experiment.py.
        """
        h_s  = self._encode(x_s)
        h_t  = self._encode(x_t)

        pred_s = self.regressor(h_s)
        oc_s   = self.oc_head(h_s)


        d_s = self.domain_disc(self._apply_grl(h_s))
        d_t = self.domain_disc(self._apply_grl(h_t))

        return pred_s, h_s, oc_s, d_s, d_t
