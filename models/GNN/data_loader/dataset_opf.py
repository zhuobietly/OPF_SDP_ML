import torch
from torch.utils.data import Dataset
"""
The DataLoader first decides which indices to fetch (e.g. [7, 123, 59, ...]).
For each index i, it calls train_ds.__getitem__(i) to get a single sample.
These individual samples are then passed to collate_fn, which combines them into a batch (usually a dict or tuple).
The batch is returned to you, so the batch is the input for the whole iteration.
"""

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
        N = raw["A"].shape[0]
        A = raw["A"]
        I = torch.eye(N)
        D_inv_sqrt = torch.diag((A.sum(dim=1)+1e-8).pow(-0.5))
        A_hat = D_inv_sqrt @ (A + I) @ D_inv_sqrt
        y_reg = torch.as_tensor(raw.get("y_reg", raw.get("y")), dtype=torch.float32).view(-1)  # (K,)
        y_cls = torch.argmin(y_reg).long()
        data = {"A_hat": A_hat.float(), "X": X.float(),
            "y_reg": y_reg,     
            "y_cls": y_cls
        }
        if self.has_global:
            gvec = torch.as_tensor(raw["global_vec"], dtype=torch.float32).view(-1)
            data["gvec"] = self.normalizer.transform(gvec)
        return data
    
def make_collate_fn(dataset):
    def collate_fn(batch):
        A = torch.stack([b["A_hat"] for b in batch], dim=0)
        X = torch.stack([b["X"] for b in batch], dim=0)
        y_reg = torch.stack([b["y_reg"].view(-1) for b in batch], dim=0)
        y_cls = torch.stack([b["y_cls"] for b in batch], dim=0)
        out = {
                "A_hat": torch.stack([b["A_hat"] for b in batch], 0),
                "X":     torch.stack([b["X"]     for b in batch], 0),
                "y_reg": torch.stack([b["y_reg"].view(-1) for b in batch], 0),
                "y_cls": torch.stack([b["y_cls"] for b in batch], 0),
            }
        if dataset.has_global:
                out["gvec"] = torch.stack([b["gvec"] for b in batch], 0)
        return out
    return collate_fn