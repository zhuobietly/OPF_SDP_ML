import torch
import torch.nn as nn
from .registry import register
from .base import GNNBase
import logging
class GraphConv(nn.Module):
    def __init__(self, in_features:int, out_features:int,
                 dropout:float=0.0, activ:str="relu"):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU() if activ == "relu" else nn.Tanh()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.step_count = 0  # 用于记录步数

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
    -> pooling: [B, Fk+G]
    -> concat gvec: [B, Fk+G]
    -> MLP -> [B, out_dim]
    """
    def __init__(self, in_dim, hidden, out_dim, g_dim, dropout=0.0, readout="mean"):
        super().__init__()
        fin = in_dim
        layers = []
        for h in hidden:
            layers.append(GraphConv(fin, h, dropout))
            fin = h  # 最终 fin = hidden[-1]
        
        self.gnn = nn.ModuleList(layers)
        self.readout = readout
        
        # 修复：确保维度一致
        final_gnn_dim = hidden[-1] if hidden else in_dim  # GNN最后输出维度
        
        self.mlp = nn.Sequential(
            nn.Linear(final_gnn_dim + g_dim, (final_gnn_dim + g_dim)//2),
            nn.ReLU(),
            nn.Linear((final_gnn_dim + g_dim)//2, out_dim),
        )
        self.lastmlp = nn.Linear(out_dim, 1)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.step_count = 0  # 用于记录步数
    
    def _readout(self, H):
        if self.readout == "mean": return H.mean(1)
        if self.readout == "sum":  return H.sum(1)
        if self.readout == "max":  return H.max(1).values
        raise ValueError("bad readout")
    def forward(self, A_hat, X, gvec):
        H = X                               # [B, N, in_dim]
        for layer in self.gnn:
            H = layer(H, A_hat)             # [B, N, hidden[-1]]
        
        h = self._readout(H)                # [B, hidden[-1]]
        z = torch.cat([h, gvec], dim=-1)    # [B, hidden[-1] + g_dim]

        out = self.mlp(z)                    # [B, out_dim] ✅ 维度匹配
        out_2 = self.lastmlp(out)            # [B, 1]
        out_3 = torch.argmin(out, dim=1)
        # 这里写个代码把每一代的out和out_2都log记下来行吗
        self.step_count += 1
        self.logger.info(f"Step {self.step_count} - out shape: {out}, out_2 shape: {out_2},out_3 shape: {out_3}")
        # return out, out_3
        return out, out_2, out_3  