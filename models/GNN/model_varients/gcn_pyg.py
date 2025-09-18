import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from .registry import register

@register("GCN_PYG")
class GCN_PYG(nn.Module):
    """
    PyG 版本的 GCN：
      - 输入: Data/Batch (含 x, edge_index, edge_weight, batch)
      - 读出: global_mean_pool（图级回归/分类都可）
    """
    def __init__(self, in_dim, hidden=[128, 128], out_dim=1, dropout=0.2):
        super().__init__()
        chs = [in_dim] + list(hidden)
        self.convs = nn.ModuleList([GCNConv(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(chs[-1], out_dim)

    def forward(self, data):
        # data: PyG 的 Batch，自动合并了多张不等尺寸图
        x, ei, ew, b = data.x, data.edge_index, getattr(data, "edge_weight", None), data.batch
        for conv in self.convs:
            x = self.drop(self.act(conv(x, ei, edge_weight=ew)))
        x = global_mean_pool(x, b)   # (num_graphs, hidden)
        return self.head(x)          # (num_graphs, out_dim)
