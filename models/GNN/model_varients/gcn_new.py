import torch, torch.nn as nn
from registries import register

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)
    def forward(self, A_hat, X):
        H = torch.matmul(A_hat, X)           # (B,N,F)
        H = self.lin(H)
        return self.drop(self.act(self.ln(H)))

@register("model", "gcn_basic")
class GCNBasic(nn.Module):
    """
    统一接口：只吃 batch，按需要返回不同预测键。
      必需输入:  batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
      可选输出:  pred_y_reg (B,), pred_arr_reg (B,K), logits(B,C)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",out_array_dim=None, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden]*layers
        self.gcns = nn.ModuleList([GraphConv(dims[i], dims[i+1], dropout) for i in range(layers)])
        self.readout = readout
        fuse_dim = hidden
        self.head_array  = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None 
        
    def _readout(self, H):
        if self.readout == "sum": return H.sum(1)
        if self.readout == "max": return H.max(1).values
        return H.mean(1)

    def forward(self, batch):
        A_hat, X = batch["A_hat"], batch["X"]
        H = X
        for g in self.gcns:
            H = g(A_hat, H)                  # (B,N,hidden)
        gcnout = self._readout(H)              # (B,hidden)

        out = {}
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
        return out

@register("model", "gcn_global")
class GCNGlobal(nn.Module):
    """
    统一接口：只吃 batch，按需要返回不同预测键。
      必需输入:  batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
      可选输出:  pred_y_reg (B,), pred_arr_reg (B,K), logits(B,C)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",
                 out_scalar=True, out_array_dim=None, num_classes=None, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden]*layers
        self.gcns = nn.ModuleList([GraphConv(dims[i], dims[i+1], dropout) for i in range(layers)])
        self.readout = readout
        fuse_dim = hidden
        self.head_scalar = nn.Linear(fuse_dim, 1) if out_scalar else None
        self.head_array  = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None
        self.head_cls    = nn.Linear(fuse_dim, num_classes) if num_classes else None

    def _readout(self, H):
        if self.readout == "sum": return H.sum(1)
        if self.readout == "max": return H.max(1).values
        return H.mean(1)

    def forward(self, batch):
        A_hat, X = batch["A_hat"], batch["X"]
        H = X
        for g in self.gcns:
            H = g(A_hat, H)                  # (B,N,hidden)
        gcnout = self._readout(H)              # (B,hidden)

        out = {}
        if self.head_scalar is not None:
            out["pred_y_reg"] = self.head_scalar(gcnout).squeeze(-1)
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
        if self.head_cls is not None:
            out["logits"] = self.head_cls(gcnout)
        return out
