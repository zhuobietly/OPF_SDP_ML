# dataset_opf.py
import torch
from torch.utils.data import Dataset
import numpy as np
from gcn_utils import normalize

def edge_to_dense_A(edge_index, edge_weight, N, dtype=torch.float32, device="cpu"):
    """
    将 numpy (edge_index, edge_weight) 转换为稠密 A:(N,N)
    """
    edge_index = torch.from_numpy(edge_index).long().to(device)
    edge_weight = torch.from_numpy(edge_weight).to(dtype=dtype, device=device)
    A_sp = torch.sparse_coo_tensor(edge_index, edge_weight, size=(N, N), dtype=dtype, device=device)
    return A_sp.to_dense()

def normalize_A_hat(A):
    """
    对称归一化 A_hat = D^{-1/2} (A + I) D^{-1/2}
    A: (N, N) torch.Tensor
    """
    N = A.size(0)
    I = torch.eye(N, dtype=A.dtype, device=A.device)
    A_tilde = A + I
    deg = A_tilde.sum(dim=1).clamp_min(1e-8)
    D_inv_sqrt = deg.pow(-0.5)
    return D_inv_sqrt.view(-1,1) * A_tilde * D_inv_sqrt.view(1,-1)

class OPFGraphDataset(Dataset):
    """
    - Reader 决定 samples（数量/包含哪些键）
    - Dataset 只做一件“特殊事”：把 raw['A'] 的边列表转成稠密 A，并归一化成 A_hat
    - 其他键（X、各类 y_*、extra 等）不判断名字，原样透传
    - 可选：按需要用 build_graph/build_features 生成 X（保持你原逻辑）
    """
    def __init__(self, samples, build_features=None,  device="cpu"):
        self.samples = samples
        self.build_features = build_features     
        self.device = torch.device(device)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        out = {}
        # 1) deal with the feature part
        if self.build_features is not None:
            X = self.build_features(raw)
            X = X.to(self.device).float()
            out["X"] = X
            N = X.shape[0]

        # 2) deal with the graph part
        if "A" in raw:
            edge_index, edge_weight = raw["A"]
        elif "A_grid" in raw:
            edge_index, edge_weight = raw["A_grid"]
        else:
            raise KeyError("样本缺少键 'A' 或 'A_grid'（边列表）。")
     # 允许 numpy 或 torch
        A = edge_to_dense_A(edge_index, edge_weight, N, dtype=torch.float32, device=self.device)
        A_hat = normalize_A_hat(A)
        out["A_hat"] = A_hat

        # 3) deal with the other part, y class
        for k, v in raw.items():
            if k == "A" or k == "A_grid":
                continue  # 已经转成 A_hat 了
            if k in out:
                continue  # X 已经填过
            #  they all have been converted to torch in reader
            out[k] = v.to(self.device)
        return out
def make_collate_fn(dataset):
    def collate_fn(batch):
        # 所有值都是 tensor，直接 stack
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            out[k] = torch.stack(vals, 0)
        return out
    return collate_fn
