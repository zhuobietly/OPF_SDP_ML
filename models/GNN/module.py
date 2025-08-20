# gnn_basic.py
from __future__ import annotations
import os, time, json, math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- Utils: device & seed ----------------
def _select_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

# ---------------- Dataset (A_hat, X, y) ---------------
class GraphSamplesDataset(Dataset):
    """
    每个样本：A_hat (N,N), X (N,F), y (K,) —— 全图回归/评分。
    A_hat 与 X 为 numpy.ndarray（float32/float64），内部转 float32 tensor。
    """
    def __init__(self, A_list: Sequence[np.ndarray], X_list: Sequence[np.ndarray], Y: np.ndarray):
        assert len(A_list) == len(X_list) == len(Y), "长度不一致"
        self.A_list = A_list
        self.X_list = X_list
        self.Y = Y.astype(np.float32)

        # 基本一致性检查（可不同 N/F，但通常相同）
        N0 = X_list[0].shape[0]
        assert all(x.shape[0] == N0 for x in X_list), "样本间节点数 N 不一致"

    def __len__(self): return len(self.X_list)

    def __getitem__(self, idx: int):
        A = torch.from_numpy(self.A_list[idx]).float()   # (N,N)
        X = torch.from_numpy(self.X_list[idx]).float()   # (N,F)
        y = torch.from_numpy(self.Y[idx]).float()        # (K,)
        return A, X, y

def _collate_graphs(batch):
    """
    将一批 (A,X,y) 叠成：
      A: (B,N,N), X: (B,N,F), y: (B,K)
    要求该 batch 中 N 一致（你的 case14/118 满足）。
    """
    A, X, y = zip(*batch)
    A = torch.stack(A, dim=0)
    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)
    return A, X, y

# ---------------- GraphConv block ---------------------
class GraphConv(nn.Module):
    """
    简单 GCN：H' = act( Dropout( A_hat @ H ) W + b )
    - 这里 A_hat 由 batch 给定（B,N,N），不在模块内部存。
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, activ: str = "relu"):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU() if activ == "relu" else nn.Tanh()

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # H: (B,N,F_in), A_hat: (B,N,N)
        H_agg = torch.bmm(A_hat, H)      # (B,N,F_in)
        H_agg = self.drop(H_agg)
        return self.act(self.lin(H_agg)) # (B,N,F_out)

# ---------------- GCN model --------------------------
class GCNBasic(nn.Module):
    """
    堆叠 L 个 GraphConv，然后全图 mean pooling → 线性头输出 K。
    """
    def __init__(self, in_features: int, hidden: List[int], out_dim: int,
                 dropout: float = 0.0, activ: str = "relu"):
        super().__init__()
        layers: List[nn.Module] = []
        fin = in_features
        for h in hidden:
            layers.append(GraphConv(fin, h, dropout=dropout, activ=activ))
            fin = h
        self.convs = nn.ModuleList(layers)
        self.head = nn.Linear(fin, out_dim)

    def forward(self, A_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # A_hat: (B,N,N), X: (B,N,F)
        H = X
        for conv in self.convs:
            H = conv(H, A_hat)           # (B,N,h)
        H_graph = H.mean(dim=1)          # global mean pooling → (B,h)
        return self.head(H_graph)        # (B,K)

# ----------------- Trainer wrapper -------------------
@dataclass
class _History:
    train: List[float]
    val:   List[float]

class GCNBasicRegressorTorch:
    """
    纯 GCN 回归/打分器（图级输出）。
    - X 来自 DataFrame['gnn_X'] (N,F)
    - A_hat 来自 DataFrame['gnn_A_hat'] (N,N，已归一化)
    - Y 来自你选定的 chordal 列 (K)

    提供：
      - fit(train_set, val_set)
      - predict(dataset) -> ndarray (num_samples, K)
      - predict_argmin(dataset) -> ndarray (num_samples,)
      - plot_learning_curves()
    """
    def __init__(self,
                 in_features: int,
                 hidden: List[int],
                 out_dim: int,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 max_epochs: int = 200,
                 early_stopping_patience: int = 20,
                 early_stopping_metric: str = "val_mse",   # or "val_regret"
                 grad_clip: Optional[float] = None,
                 device: str = "auto",
                 save_dir: Optional[str] = None,
                 random_seed: int = 42):
        _set_seed(random_seed)
        self.device = _select_device(device)
        self.model = GCNBasic(in_features, hidden, out_dim, dropout=0.0).to(self.device)
        self.criterion = nn.MSELoss(reduction="mean")
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.grad_clip = grad_clip
        self.hist = _History(train=[], val=[])
        self.save_dir = save_dir or os.path.join("runs", f"gcn_basic_{_timestamp()}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.out_dim = out_dim
        self.random_seed = random_seed

    # ------------- high-level fit ---------------------
    def fit(self, train_set: GraphSamplesDataset, val_set: GraphSamplesDataset, verbose: bool = True):
        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  shuffle=True, drop_last=False, collate_fn=_collate_graphs)
        val_loader   = DataLoader(val_set, batch_size=self.batch_size,
                                  shuffle=False, drop_last=False, collate_fn=_collate_graphs)

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(1, self.max_epochs + 1):
            # ---- train ----
            self.model.train()
            losses = []
            for A, X, y in train_loader:
                A, X, y = A.to(self.device), X.to(self.device), y.to(self.device)
                self.optim.zero_grad(set_to_none=True)
                y_hat = self.model(A, X)
                loss = self.criterion(y_hat, y)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
                self.optim.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses)) if losses else float("nan")

            # ---- validate ----
            self.model.eval()
            with torch.no_grad():
                v_losses, v_regrets = [], []
                for A, X, y in val_loader:
                    A, X, y = A.to(self.device), X.to(self.device), y.to(self.device)
                    y_hat = self.model(A, X)
                    v_losses.append(float(self.criterion(y_hat, y).cpu()))
                    # regret 评价（按 row 最小列的真实值平均）
                    preds = torch.argmin(y_hat, dim=1)
                    v_regrets.append(torch.mean(y[torch.arange(y.size(0)), preds]).item())
                val_mse = float(np.mean(v_losses)) if v_losses else float("nan")
                val_regret = float(np.mean(v_regrets)) if v_regrets else float("nan")
                monitor = val_regret if self.early_stopping_metric == "val_regret" else val_mse

            self.hist.train.append(train_loss)
            self.hist.val.append(monitor)
            if verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}  "
                      f"val_{'regret' if self.early_stopping_metric=='val_regret' else 'mse'}={monitor:.6f}")

            # ---- early stopping ----
            if monitor + 1e-12 < best_val:
                best_val = monitor
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}. Best val={best_val:.6f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._save_config()
        return self

    @torch.no_grad()
    def predict(self, dataset: GraphSamplesDataset, batch_size: int = 64) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_graphs)
        self.model.eval()
        outs: List[np.ndarray] = []
        for A, X, _ in loader:
            A, X = A.to(self.device), X.to(self.device)
            Y_hat = self.model(A, X).cpu().numpy()
            outs.append(Y_hat)
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def predict_argmin(self, dataset: GraphSamplesDataset, batch_size: int = 64) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_graphs)
        self.model.eval()
        preds: List[np.ndarray] = []
        for A, X, _ in loader:
            A, X = A.to(self.device), X.to(self.device)
            p = torch.argmin(self.model(A, X), dim=1).cpu().numpy()
            preds.append(p)
        return np.concatenate(preds, axis=0)

    def plot_learning_curves(self, save_name: str = "learning_curves.png"):
        fig = plt.figure(figsize=(6,4))
        plt.plot(self.hist.train, label="train")
        plt.plot(self.hist.val,   label="val")
        plt.xlabel("epoch")
        plt.ylabel("val_regret" if self.early_stopping_metric=="val_regret" else "val_mse")
        plt.title("GCN learning curves")
        plt.legend()
        path = os.path.join(self.save_dir, save_name)
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _save_config(self):
        cfg = dict(
            out_dim=self.out_dim,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_metric=self.early_stopping_metric,
            device=str(self.device),
            random_seed=self.random_seed,
            model="gcn_basic",
        )
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

# ----------------- Helper: from DataFrame -------------
def build_datasets_from_dataframe(df_train,
                                  df_val,
                                  y_cols: Sequence[str]) -> Tuple[GraphSamplesDataset, GraphSamplesDataset, int, int, int]:
    """
    df_* 应包含两列：
      - 'gnn_A_hat' : (N,N) ndarray（已归一化）
      - 'gnn_X'     : (N,F) ndarray
    y_cols: 目标列名（如所有包含 'Chordal' 的列）
    返回：train_set, val_set, N, F, K
    """
    # 取形状
    A0: np.ndarray = df_train["gnn_A_hat"].iloc[0]
    X0: np.ndarray = df_train["gnn_X"].iloc[0]
    N, F = X0.shape
    K = len(y_cols)

    tr_set = GraphSamplesDataset(
        df_train["gnn_A_hat"].tolist(),
        df_train["gnn_X"].tolist(),
        df_train[y_cols].to_numpy()
    )
    va_set = GraphSamplesDataset(
        df_val["gnn_A_hat"].tolist(),
        df_val["gnn_X"].tolist(),
        df_val[y_cols].to_numpy()
    )
    return tr_set, va_set, N, F, K

# --------------------- Demo ---------------------------
if __name__ == "__main__":
    # 合成一个小 demo（同 N 的多图）
    N, F, K = 14, 12, 6
    num_train, num_val = 64, 16
    A = np.eye(N, dtype=np.float32)
    A = A + (np.ones((N,N), dtype=np.float32)-(np.eye(N, dtype=np.float32)))*0.05
    A_list_tr = [A for _ in range(num_train)]
    A_list_va = [A for _ in range(num_val)]
    X_list_tr = [np.random.randn(N, F).astype(np.float32) for _ in range(num_train)]
    X_list_va = [np.random.randn(N, F).astype(np.float32) for _ in range(num_val)]
    Wtrue = np.random.randn(F, K).astype(np.float32)
    y_tr = np.array([X.mean(axis=0) @ Wtrue + 0.05*np.random.randn(K) for X in X_list_tr], dtype=np.float32)
    y_va = np.array([X.mean(axis=0) @ Wtrue + 0.05*np.random.randn(K) for X in X_list_va], dtype=np.float32)

    tr_set = GraphSamplesDataset(A_list_tr, X_list_tr, y_tr)
    va_set = GraphSamplesDataset(A_list_va, X_list_va, y_va)

    model = GCNBasicRegressorTorch(
        in_features=F, hidden=[64, 64], out_dim=K,
        lr=1e-3, weight_decay=1e-4, batch_size=16,
        max_epochs=100, early_stopping_patience=15,
        early_stopping_metric="val_mse", grad_clip=1.0,
        device="auto", random_seed=42
    )
    model.fit(tr_set, va_set, verbose=True)
    print("Curves saved to:", model.plot_learning_curves())
