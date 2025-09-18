import torch
import torch.nn as nn
from .registry import register
from .base import GNNBase

class GraphConv(nn.Module):
    def __init__(self, in_features:int, out_features:int,
                 dropout:float=0.0, activ:str="relu"):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU() if activ == "relu" else nn.Tanh()

    def forward(self, H, A_hat):
        H_agg = torch.bmm(A_hat, H)
        H_agg = self.drop(H_agg)
        return self.act(self.lin(H_agg))

@register("GCN")
class GCN(GNNBase):
    def __init__(self, in_dim:int, hidden:list[int], out_dim:int,
                 dropout:float=0.0, activ:str="relu"):
        super().__init__(in_dim, hidden, out_dim, dropout, activ)
        layers = []
        fin = in_dim
        for h in hidden:
            layers.append(GraphConv(fin, h, dropout=dropout, activ=activ))
            fin = h
        self.convs = nn.ModuleList(layers)
        self.head = nn.Linear(fin, out_dim)

    def forward(self, A_hat, X):
        H = X
        for conv in self.convs:
            H = conv(H, A_hat)
        H_mean = H.mean(dim=1)
        return self.head(H_mean)

@register("GCN_global")
class GCNGlobal(nn.Module):
    """
    X: [B, N, F0], A: [B, N, N], gvec: [B, G]
    -> H: [B, N, Fk]
    -> pooling: [B, Fk]
    -> concat gvec: [B, Fk+G]
    -> MLP -> [B, out_dim]
    """
    def __init__(self, in_dim, hidden, out_dim, g_dim, dropout=0.0, readout="mean"):
        super().__init__()
        fin = in_dim
        layers = []
        for h in hidden:
            layers.append(GraphConv(fin, h, dropout))
            fin = h
        self.gnn = nn.ModuleList(layers)
        self.readout = readout
        self.mlp = nn.Sequential(
            nn.Linear(fin + g_dim, max(64, (fin + g_dim)//2)),
            nn.ReLU(),
            nn.Linear(max(64, (fin + g_dim)//2), out_dim),
        )
    def _readout(self, H):
        if self.readout == "mean": return H.mean(1)
        if self.readout == "sum":  return H.sum(1)
        if self.readout == "max":  return H.max(1).values
        raise ValueError("bad readout")
    def forward(self, A_hat, X, gvec):
        H = X
        for layer in self.gnn:
            H = layer(H, A_hat)
        h = self._readout(H)               # [B,Hd]
        z = torch.cat([h, gvec], dim=-1)   # [B,Hd+G]
        return self.mlp(z)