import torch
from torch.utils.data import Dataset

class OPFGraphDataset(Dataset):
    def __init__(self, samples, build_graph, build_features):
        self.samples = samples
        self.build_graph = build_graph
        self.build_features = build_features

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        g = self.build_graph(raw)
        X = self.build_features(g, raw)
        N = raw["A"].shape[0]
        A = raw["A"]
        I = torch.eye(N)
        D_inv_sqrt = torch.diag((A.sum(dim=1)+1e-8).pow(-0.5))
        A_hat = D_inv_sqrt @ (A + I) @ D_inv_sqrt
        y = raw["label"]
        return {"A_hat": A_hat.float(), "X": X.float(),
                "y": torch.as_tensor(y).float()}

def collate_fn(batch):
    A = torch.stack([b["A_hat"] for b in batch], dim=0)
    X = torch.stack([b["X"] for b in batch], dim=0)
    y = torch.stack([b["y"].view(-1) for b in batch], dim=0)
    return {"A_hat": A, "X": X, "y": y}
