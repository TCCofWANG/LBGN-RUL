"""
MDAN: Mixup Domain Adaptation Network for RUL prediction.

Based on: Furqon et al., "MDAN: Multi-source Domain Adaptation Network"
https://github.com/furqon3009/MDAN

Key idea:
  A Bidirectional LSTM encoder is pretrained on the source domain, then
  adapted to the target through an intermediate "mixup domain".  At each
  step a mixed sample is constructed as:
      x_mix = λ·x_src + (1-λ)·x_tgt
      y_mix = λ·y_src + (1-λ)·ŷ_tgt      (target uses pseudo-labels from model)

  The mixing coefficient λ is updated adaptively using the Wasserstein distance
  between source and mixed features — if the mixture is already close to source
  (small W₁ distance), λ is reduced to pull it further toward the target:
      q = exp(-W₁(src,mix) / (W₁(src,mix) + W₁(tgt,mix)·τ))
      λ ← clamp(Uniform(λ-0.2, λ+0.2), 0, 1),  λ drifts toward (1-q)/T each step

Architecture:
  Encoder   : Bidirectional LSTM(input_dim, hidden_dim, n_layers=3)
  Feature   : last-step encoder output — shape (B, 2*hidden_dim)
  Regressor : FC(2*hid → hid → hid//2 → 1) + Sigmoid

Training (da_mode='mdan'):
  Phase 1 (epoch < pretrain_epochs) : source-only RMSE pre-training
  Phase 2 (epoch ≥ pretrain_epochs) : mixup training with adaptive λ
      loss = α * (MSE(pred_mix_input, y_mix) + MSE(pred_mix_feat, y_mix))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDAN_RUL(nn.Module):
    """
    Bidirectional LSTM + feature-space mixup DA for cross-domain RUL.
    forward(x, OCC=None) → (None, {'gamma': pred, 'h': features})
    """
    def __init__(self, args):
        super().__init__()
        in_c      = getattr(args, 'num_nodes',    14)
        hid       = getattr(args, 'hidden_dim',   32)
        n_layers  = getattr(args, 'lstm_n_layers', 3)
        dropout   = getattr(args, 'dropout',      0.1)
        bid       = getattr(args, 'lstm_bid',     True)
        self.hidden_dim = hid
        self.bid        = bid

        self.encoder = nn.LSTM(
            in_c, hid, n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True, bidirectional=bid,
        )
        feat_dim = hid * (2 if bid else 1)

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, hid),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid,      hid // 2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid // 2, 1),         nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → h: (B, 2*hidden_dim)"""
        out, _ = self.encoder(x)
        return out[:, -1, :]

    def forward(self, x, OCC=None):
        h    = self._encode(x)
        pred = self.regressor(h)
        return None, {'gamma': pred, 'h': h}

    def regress(self, h: torch.Tensor) -> torch.Tensor:
        """Run regressor on a pre-computed feature vector."""
        return self.regressor(h)
