

import torch
import torch.nn as nn
import torch.nn.functional as F


def ccdg_contrastive_loss(
        features: torch.Tensor,
        labels:   torch.Tensor,
        temperature: float = 0.7,
) -> torch.Tensor:

    feat = F.normalize(features, dim=1)
    lbl  = labels.contiguous().view(-1, 1)

    mask = torch.eq(lbl, lbl.T).float().to(features.device)

    dot  = torch.div(torch.matmul(feat, feat.T), temperature)
    dot  = dot - dot.max(dim=1, keepdim=True).values.detach()


    N    = feat.shape[0]
    self_mask = torch.scatter(
        torch.ones_like(mask), 1,
        torch.arange(N, device=features.device).view(-1, 1), 0)
    mask = mask * self_mask

    exp_dot = torch.exp(dot) * self_mask
    log_prob = dot - torch.log(exp_dot.sum(1, keepdim=True) + 1e-8)

    mask_sum = mask.sum(1)
    mask_sum[mask_sum == 0] = 1
    loss = -(mask * log_prob).sum(1) / mask_sum
    return loss.mean()


def occ_to_stage(occ_raw: torch.Tensor, max_life: float) -> torch.Tensor:
    occ_n = occ_raw.float() / max_life
    s = torch.zeros(occ_n.shape[0], dtype=torch.long, device=occ_n.device)
    s[occ_n > 0.15] = 1
    s[occ_n > 0.66] = 2
    return s


class CCDG_RUL(nn.Module):

    def __init__(self, args):
        super().__init__()
        in_c    = getattr(args, 'num_nodes',  14)
        hid     = getattr(args, 'hidden_dim', 32)
        dropout = getattr(args, 'dropout',    0.1)
        self.hidden_dim = hid


        self.encoder = nn.Sequential(
            nn.Conv1d(in_c, 64,  kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),

            nn.Conv1d(64,  128,  kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),

            nn.Conv1d(256, 256,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, hid), nn.ReLU(), nn.Dropout(dropout))

        self.regressor = nn.Sequential(
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid // 2, 1),   nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(self.encoder(x.permute(0, 2, 1)))

    def forward(self, x, OCC=None):
        h    = self._encode(x)
        pred = self.regressor(h)
        return None, {'gamma': pred, 'h': h}
