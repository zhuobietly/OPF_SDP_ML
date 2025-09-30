import torch
from .base import GraphBuilder

class OriginalGraph(GraphBuilder):
    def build(self, raw):
        A = raw["A"]

        # 情况 1：tuple (edge_index, edge_weight)
        if isinstance(A, tuple) and len(A) == 2:
            edge_index, edge_weight = A
            if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
                ei = edge_index
            else:
                ei = torch.as_tensor(edge_index, dtype=torch.long)
            edge_attr = torch.as_tensor(edge_weight, dtype=torch.float32)
            return {"edge_index": ei, "edge_attr": edge_attr, "global": {"kind": "original"}}

        # 情况 2：稠密邻接矩阵 (N,N)
        if isinstance(A, torch.Tensor) and A.dim() == 2:
            rows, cols = A.nonzero(as_tuple=True)
            ei = torch.stack([rows, cols], dim=0)  # [2,E]
            # 如果你想保留权重, 取消下一行注释; 不要权重就沿用 None
            # edge_attr = A[rows, cols].to(torch.float32)
            return {"edge_index": ei.contiguous(), "edge_attr": None, "global": {"kind": "original"}}

        # 情况 3：已经是 edge_index
        if isinstance(A, torch.Tensor):
            # 期望 A 形状就是 [2,E]
            return {"edge_index": A, "edge_attr": None, "global": {"kind": "original"}}

        # 兜底：numpy 等
        import numpy as np
        if isinstance(A, np.ndarray) and A.ndim == 2:
            rows, cols = (A != 0).nonzero()
            ei = torch.tensor([rows, cols], dtype=torch.long)
            return {"edge_index": ei, "edge_attr": None, "global": {"kind": "original"}}

        raise TypeError("Unsupported type for raw['A']")
