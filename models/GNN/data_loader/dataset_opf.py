import torch
from torch.utils.data import Dataset
import numpy as np
"""
The DataLoader first decides which indices to fetch (e.g. [7, 123, 59, ...]).
For each index i, it calls train_ds.__getitem__(i) to get a single sample.
These individual samples are then passed to collate_fn, which combines them into a batch (usually a dict or tuple).
The batch is returned to you, so the batch is the input for the whole iteration.
"""
def edge_to_A(edge_index, edge_weight, N):
    """
    输入:
      - edge_index: (2,E) int32
      - edge_weight: (E,) float32
      - N: number of nodes
    输出:
      - A: (N,N) float32 numpy.ndarray
    """
    A = np.zeros((N, N), dtype=np.float32)
    rows, cols = edge_index
    A[rows, cols] = edge_weight
    return A

class OPFGraphDataset(Dataset):
    def __init__(self, samples, build_graph, build_features, norm):
        self.samples = samples
        self.build_graph = build_graph
        self.build_features = build_features
        self.has_global = "global_vec" in samples[0]
        self.normalizer = norm 

    def __len__(self): return len(self.samples)
    
    

    def __getitem__(self, idx):
        raw = self.samples[idx]
        g = self.build_graph(raw)
        # g did not used now,X .shape = torch.Size([N, F])[28,3]
        X = self.build_features(g, raw)
        N = X.shape[0]
        (edge_index, edge_weight) = raw["A"]
        A = edge_to_A(edge_index, edge_weight, N)
        A = torch.from_numpy(A).to(dtype=torch.float32)
        I = torch.eye(N, dtype=A.dtype, device=A.device)
        deg = (A + I).sum(dim=1).clamp_min(1e-8)
        D = torch.pow(deg, -0.5)
        A_hat = D.view(-1,1) * (A + I) * D.view(1,-1)
        y_reg = torch.as_tensor(raw.get("y_reg"), dtype=torch.float32).view(-1)  # (K,)
        y_cls = torch.as_tensor(raw.get("y_cls"), dtype=torch.long)
        y_arr_reg = torch.as_tensor(raw.get("y_arr_reg"), dtype=torch.float32).view(-1)  # (K,)
        
        data = {"A_hat": A_hat.float(), "X": X.float(),
            "y_reg": y_reg,
            "y_cls": y_cls,
            "y_arr_reg": y_arr_reg
        }
        if self.has_global:
            gvec = torch.as_tensor(raw["global_vec"], dtype=torch.float32).view(-1)
            data["gvec"] = self.normalizer.transform(gvec)
        return data
    
def make_collate_fn(dataset):
    def collate_fn(batch):
        out = {
                "A_hat": torch.stack([b["A_hat"] for b in batch], 0),
                "X":     torch.stack([b["X"]     for b in batch], 0),
                "y_reg": torch.stack([b["y_reg"].view(-1) for b in batch], 0),
                "y_cls": torch.stack([b["y_cls"] for b in batch], 0),
                "y_arr_reg": torch.stack([b["y_arr_reg"].view(-1) for b in batch], 0),
            }
        if dataset.has_global:
                out["gvec"] = torch.stack([b["gvec"] for b in batch], 0)
        return out
    return collate_fn