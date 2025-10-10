import torch
import torch.nn as nn
from .registry import register
from .base import GNNBase
import logging
class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.0, activ: str = "relu"):
        super().__init__()
        self.lin  = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act  = nn.ReLU(inplace=True) if activ == "relu" else nn.Tanh()
        self.ln   = nn.LayerNorm(out_features)
        # 当维度不一致时，用线性投影对齐做残差
        self.proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, H, A_hat):
        # 邻居聚合
        H_agg = torch.bmm(A_hat, H)              # [B, N, Fin]
        # 线性 + Dropout + 激活
        H_new = self.act(self.drop(self.lin(H_agg)))
        # 残差 + LayerNorm
        H_res = self.proj(H)
        return self.ln(H_new + H_res)

# @register("GCN_global")
# class GCNGlobal(nn.Module):
#     """
#     X: [B, N, F0], A: [B, N, N], gvec: [B, G]
#     -> H: [B, N, Fk]
#     -> pooling: [B, Fk+G]
#     -> concat gvec: [B, Fk+G]
#     -> MLP -> [B, out_dim]
#     """
#     def __init__(self, in_dim, hidden, out_dim, g_dim, dropout=0.0, readout="mean"):
#         super().__init__()
#         fin = in_dim
#         layers = []
#         for h in hidden:
#             layers.append(GraphConv(fin, h, dropout))
#             fin = h  # 最终 fin = hidden[-1]
        
#         self.gnn = nn.ModuleList(layers)
#         self.readout = readout
        
#         # 修复：确保维度一致
#         final_gnn_dim = hidden[-1] if hidden else in_dim  # GNN最后输出维度
        
#         self.mlp = nn.Sequential(
#             nn.Linear(final_gnn_dim + g_dim, (final_gnn_dim + g_dim)//2),
#             nn.ReLU(),
#             nn.Linear((final_gnn_dim + g_dim)//2, out_dim),
#         )
#         self.lastmlp = nn.Linear(out_dim, 1)
        
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.step_count = 0  # 用于记录步数
    
#     def _readout(self, H):
#         if self.readout == "mean": return H.mean(1)
#         if self.readout == "sum":  return H.sum(1)
#         if self.readout == "max":  return H.max(1).values
#         raise ValueError("bad readout")
#     def forward(self, A_hat, X, gvec):
#         H = X                               # [B, N, in_dim]
#         for layer in self.gnn:
#             H = layer(H, A_hat)             # [B, N, hidden[-1]]
        
#         h = self._readout(H)                # [B, hidden[-1]]
#         z = torch.cat([h, gvec], dim=-1)    # [B, hidden[-1] + g_dim]

#         out = self.mlp(z)                    # [B, out_dim] ✅ 维度匹配
#         out_2 = self.lastmlp(out)            # [B, 1]
#         out_3 = torch.argmin(out, dim=1)
#         # 这里写个代码把每一代的out和out_2都log记下来行吗
#         self.step_count += 1
#         self.logger.info(f"Step {self.step_count} - out shape: {out}, out_2 shape: {out_2},out_3 shape: {out_3}")
#         # return out, out_3
#         return out, out_2, out_3  
@register("GCN")
class GCN(nn.Module):
    """
    X: [B, N, F0], A: [B, N, N], gvec: [B, G]
    返回: (out_24, out_scalar, argmin(out_24))  —— 与你原接口一致
    """
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0, readout="sum"):
        super().__init__()
        fin = in_dim
        layers = []
        for h in hidden:
            layers.append(GraphConv(fin, h, dropout))
            fin = h                               # fin = hidden[-1]
        self.gnn = nn.ModuleList(layers)

        self.readout = readout
        final_gnn_dim = hidden[-1] if hidden else in_dim

        # === 全局注意力池化（1-head）===
        # self.att_q = nn.Linear(final_gnn_dim, 1)

        # === 旧的 24 维分支（保持兼容；如无用也可保留）===
        # self.mlp_24 = nn.Sequential(
        #     nn.Linear(final_gnn_dim , (final_gnn_dim )//2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear((final_gnn_dim )//2, out_dim),
        # )

        # === 新的“标量回归专用”分支（不经过 24 维瓶颈）===
        self.scalar_head = nn.Sequential(
            nn.Linear(final_gnn_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)            # 注意：最终线性，无激活
        )

        # self.logger = logging.getLogger(self.__class__.__name__)
        self.step_count = 0
    
        z_dim = final_gnn_dim    # [B, F+G]

        self.z_norm = nn.LayerNorm(z_dim)


    def _readout(self, H):
        if self.readout == "mean":
            return H.mean(1)
        if self.readout == "sum":
            return H.sum(1)
        if self.readout in ("attn", "attention"):
            score = self.att_q(H).squeeze(-1)     # [B, N]
            alpha = torch.softmax(score, dim=1)   # [B, N]
            return (alpha.unsqueeze(-1) * H).sum(dim=1)  # [B, F]
        raise ValueError(f"bad readout: {self.readout}")
    

    def forward(self, A_hat, X, gvec=None):
        # GNN
        H = X
        for layer in self.gnn:
            H = layer(H, A_hat)                   # [B, N, F]

        # Readout
        h = self._readout(H)                      # [B, F]
        if gvec is not None:
            z = torch.cat([h], dim=-1)      # [B, F+G]
        else:
            z = h

        # 两个分支           # self.z_norm = nn.LayerNorm(z_dim)
        s = self.scalar_head(z)         # 线性头输出到实数
 
        # out_24 = self.mlp_24(z)             # [B, out_dim]  (如果你保留了 24 维分支)
        # out_scalar = self.scalar_head(z)          # [B, 1]
        #这个不可导也没用到
        # with torch.no_grad():
        #     argmin_idx = torch.argmin(out_24, dim=1)  

        # # 轻量日志：范围与均值
        # if (self.step_count % 50) == 0:           # 每 50 步打印一次
        #     with torch.no_grad():
        #         p = out_scalar.squeeze(-1)
        #         self.logger.info(
        #             f"[step {self.step_count}] scalar pred "
        #             f"min={p.min().item():.4f} max={p.max().item():.4f} mean={p.mean().item():.4f}"
        # #         )
        # self.step_count += 1
        return s

@register("GCN_global")
class GCNGlobal(nn.Module):
    """
    X: [B, N, F0], A: [B, N, N], gvec: [B, G]
    返回: (out_24, out_scalar, argmin(out_24))  —— 与你原接口一致
    """
    def __init__(self, in_dim, hidden, out_dim, g_dim, dropout=0.0, readout="sum"):
        super().__init__()
        fin = in_dim
        layers = []
        for h in hidden:
            layers.append(GraphConv(fin, h, dropout))
            fin = h                               # fin = hidden[-1]
        self.gnn = nn.ModuleList(layers)

        self.readout = readout
        final_gnn_dim = hidden[-1] if hidden else in_dim

        # === 全局注意力池化（1-head）===
        self.att_q = nn.Linear(final_gnn_dim, 1)

        # === 旧的 24 维分支（保持兼容；如无用也可保留）===
        self.mlp_24 = nn.Sequential(
            nn.Linear(final_gnn_dim + g_dim, (final_gnn_dim + g_dim)//2),
            nn.ReLU(inplace=True),
            nn.Linear((final_gnn_dim + g_dim)//2, out_dim),
        )

        # === 新的“标量回归专用”分支（不经过 24 维瓶颈）===
        self.scalar_head = nn.Sequential(
            nn.Linear(final_gnn_dim + g_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)            # 注意：最终线性，无激活
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.step_count = 0
    
        z_dim = final_gnn_dim + g_dim     # [B, F+G]

        self.z_norm = nn.LayerNorm(z_dim)


    def _readout(self, H):
        if self.readout == "mean":
            return H.mean(1)
        if self.readout == "sum":
            return H.sum(1)
        if self.readout in ("attn", "attention"):
            score = self.att_q(H).squeeze(-1)     # [B, N]
            alpha = torch.softmax(score, dim=1)   # [B, N]
            return (alpha.unsqueeze(-1) * H).sum(dim=1)  # [B, F]
        raise ValueError(f"bad readout: {self.readout}")
    

    def forward(self, A_hat, X, gvec=None):
        # GNN
        H = X
        for layer in self.gnn:
            H = layer(H, A_hat)                   # [B, N, F]

        # Readout + 拼接 gvec
        h = self._readout(H)                      # [B, F]
        if gvec is not None:
            z = torch.cat([h, gvec], dim=-1)      # [B, F+G]
        else:
            z = h

        # 两个分支           # self.z_norm = nn.LayerNorm(z_dim)
        s = self.scalar_head(z)         # 线性头输出到实数
        y_hat = 2.0 * torch.sigmoid(s) 
        out_24 = self.mlp_24(z)             # [B, out_dim]  (如果你保留了 24 维分支)
        out_scalar = self.scalar_head(z)          # [B, 1]
        #这个不可导也没用到
        with torch.no_grad():
            argmin_idx = torch.argmin(out_24, dim=1)  

        # 轻量日志：范围与均值
        if (self.step_count % 50) == 0:           # 每 50 步打印一次
            with torch.no_grad():
                p = out_scalar.squeeze(-1)
                self.logger.info(
                    f"[step {self.step_count}] scalar pred "
                    f"min={p.min().item():.4f} max={p.max().item():.4f} mean={p.mean().item():.4f}"
                )
        self.step_count += 1
        return out_24, y_hat, argmin_idx
