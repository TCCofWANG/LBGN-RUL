

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class ChebConvDense(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.K = K

        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=(k == 0))
            for k in range(K)
        ])

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:

        Tx_0 = x
        out   = self.lins[0](Tx_0)

        if self.K > 1:
            Tx_1 = L @ x
            out  = out + self.lins[1](Tx_1)
            for k in range(2, self.K):
                Tx_2 = 2.0 * (L @ Tx_1) - Tx_0
                out  = out + self.lins[k](Tx_2)
                Tx_0, Tx_1 = Tx_1, Tx_2

        return out


def _sym_norm_adjacency(A: torch.Tensor) -> torch.Tensor:

    A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    D_inv_sqrt = A.sum(dim=1).clamp(min=1e-8).pow(-0.5)
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


class GraphGenerationLayer(nn.Module):


    def __init__(self, in_features: int = 256, attr_dim: int = 10):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, attr_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        attr = self.proj(x)
        A    = torch.mm(attr, attr.T)
        row_max = A.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        return A / row_max


class MultiReceptiveFieldGCN(nn.Module):


    def __init__(self, in_channels: int = 256):
        super().__init__()

        self.scale1_K1 = ChebConvDense(in_channels, 400, K=1)
        self.scale1_K2 = ChebConvDense(in_channels, 400, K=2)
        self.scale1_K3 = ChebConvDense(in_channels, 400, K=3)
        self.bn1        = nn.BatchNorm1d(1200)


        self.scale2_K1 = ChebConvDense(1200, 100, K=1)
        self.scale2_K2 = ChebConvDense(1200, 100, K=2)
        self.scale2_K3 = ChebConvDense(1200, 100, K=3)
        self.bn2        = nn.BatchNorm1d(300)

        self.fc = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:


        h = torch.cat([self.scale1_K1(x, L),
                        self.scale1_K2(x, L),
                        self.scale1_K3(x, L)], dim=-1)
        h = F.relu(self.bn1(h), inplace=True)


        h = torch.cat([self.scale2_K1(h, L),
                        self.scale2_K2(h, L),
                        self.scale2_K3(h, L)], dim=-1)
        h = F.relu(self.bn2(h), inplace=True)

        return self.fc(h)


class CNN_RUL(nn.Module):


    def __init__(self, in_channels: int = 14):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        return self.fc(x)


class DAGCN_RUL(nn.Module):


    def __init__(self, args):
        super().__init__()
        in_channels = getattr(args, 'num_nodes', 14)

        self.cnn     = CNN_RUL(in_channels)
        self.ggl     = GraphGenerationLayer(in_features=256, attr_dim=10)
        self.mrf_gcn = MultiReceptiveFieldGCN(in_channels=256)
        self.dropout  = nn.Dropout(p=getattr(args, 'dropout', 0.3))
        self.regressor = nn.Linear(256, 1)

    def forward(self, x_in: torch.Tensor, OCC=None):

        h_cnn = self.cnn(x_in)


        A = self.ggl(h_cnn)
        L = _sym_norm_adjacency(A)

        h = self.mrf_gcn(h_cnn, L)
        h = self.dropout(h)

        pred = F.softplus(self.regressor(h))

        return None, {'gamma': pred, 'h': h}
