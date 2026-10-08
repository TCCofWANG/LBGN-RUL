

import numpy as np
import torch
from torch import nn


def _grl_coefficient(iter_num: int,
                     high: float = 1.0,
                     low: float = 0.0,
                     alpha: float = 10.0,
                     max_iter: float = 10_000.0) -> float:
        return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
                 - (high - low) + low)


def _grl_hook(coeff: float):

    def _hook(grad):
        return -coeff * grad.clone()
    return _hook


class AdversarialNet(nn.Module):


    def __init__(self, in_feature: int, hidden_size: int, max_iter: float = 10_000.0):
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.iter_num = 0
        self.max_iter = max_iter
        self.alpha    = 10.0
        self.low      = 0.0
        self.high     = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.iter_num += 1
        coeff = _grl_coefficient(self.iter_num, self.high, self.low,
                                 self.alpha, self.max_iter)

        x = x * 1.0
        x.register_hook(_grl_hook(coeff))

        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.sigmoid(self.output(x))
