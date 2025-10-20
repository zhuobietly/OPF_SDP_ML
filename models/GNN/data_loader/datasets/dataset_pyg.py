import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def dense_to_edge_index(A: torch.Tensor):
    # A: (N,N) 0/1 稠密邻接；返回 edge_index:(2,E)
    idx = A.nonzero(as_tuple=False).t().contiguous()
    return idx

def gcn_norm_edge_weight(A: torch.Tensor):
    """
    为 GCN 构建对称归一化权重：A_tilde = A + I
    w_ij = 1/sqrt(d_i) * 1/sqrt(d_j)，d 是 A_tilde 的度
    """
    N = A.shape[0]
    A = A.float()
    A_t = A.clone()
    A_t.fill_diagonal_(1.0)
    deg = A_t.sum(dim=1)
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
    edge_index = dense_to_edge_index(A_t)
    w = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
    return edge_index, w

class OPFGraphDatasetPyG(Dataset):
    """
    输入：samples（raw 字典列表）、build_graph（构图函数）、build_features（特征函数）
    输出：torch_geometric.data.Data（支持不等 N）
    raw 要包含：
      - "A": (N,N) 稠密邻接（0/1）；若你有 edge_index 也行，自行改造这里
      - "label": 标量或向量（图级标签）
      - 你的特征来源（如 "node_load" 等），由 build_features 使用
    """
    def __init__(self, samples, build_graph, build_features):
        self.samples = samples
        self.build_graph = build_graph
        self.build_features = build_features

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        g = self.build_graph(raw)                    # 用不到也行，给 features 用
        x = self.build_features(g, raw).float()      # (N, F)
        A = raw["A"].float()                         # (N, N)
        edge_index, edge_weight = gcn_norm_edge_weight(A)
        y = torch.as_tensor(raw["label"]).float().view(-1)  # 图级标签
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        return data
