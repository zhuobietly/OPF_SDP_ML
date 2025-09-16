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
