"""
file: pytorch_mlp_regressor.py

A PyTorch-based reimplementation of your multi-output regressor with nearly the
same API as your XGBoost wrapper, but designed for easy "second development":
- clean forward/backward (with optional custom autograd for Linear)
- MSE loss by default (Ŷ vs Y), early stopping, learning-curve plotting
- predict() returns (N, K), predict_argmin() returns argmin over K
- mean regret metric helper (compatible with your current evaluation)
- save_dir for logs/plots just like before
- easy path to GNN with a GraphConv layer (Kipf-Welling style) included

Usage (drop-in vibe):

    from pytorch_mlp_regressor import MultiTargetMLPRegressorTorch

    model = MultiTargetMLPRegressorTorch(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        hidden_layers=[256,128,64],
        dropout=0.10,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=200,
        early_stopping_patience=20,
        early_stopping_metric="val_mse",   # or "val_regret"
        linear_impl="manual",              # "manual" uses custom backward; "autograd" uses nn.Linear
        device="auto",
        random_seed=42,
    )
    model.fit(X_train, Y_train, X_val, Y_val, verbose=True)
    Y_pred = model.predict(X_test)
    preds  = model.predict_argmin(X_test)

    from pytorch_mlp_regressor import mean_test_regret
    print("Mean test regret:", mean_test_regret(Y_test, preds))
    print("Mean test regret per class:", Y_test.mean(axis=0))
    model.plot_learning_curves()

"""
from __future__ import annotations
import os
import time
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------
# Reproducibility & device handling
# ---------------------------------
def _select_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
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


# --------------------------
# Custom autograd Linear op
# --------------------------
class _CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        # weight: (out_features, in_features); x: (N, in_features)
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.tensor([]))
        out = x.matmul(weight.t())
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_out.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_out.t().matmul(x)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out.sum(dim=0) if bias.numel() > 0 else None
        return grad_x, grad_weight, grad_bias


class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: str = "he"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters(init)

    def reset_parameters(self, init: str = "he"):
        if init == "he":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif init == "xavier":
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.uniform_(self.weight, -0.02, 0.02)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _CustomLinearFunction.apply(x, self.weight, self.bias)


# ---------------------
# Dataloader and Losses
# ---------------------
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 2 and Y.ndim == 2
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# --------------
# Core MLP model
# --------------
class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int],
                 activ: str = "relu", dropout: float = 0.0, linear_impl: str = "manual"):
        super().__init__()
        Act = nn.ReLU if activ == "relu" else nn.Tanh
        Lin = CustomLinear if linear_impl == "manual" else nn.Linear
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers += [Lin(prev, h), Act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [Lin(prev, output_dim)]  # linear head
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _History:
    train_loss: List[float]
    val_loss: List[float]


class MultiTargetMLPRegressorTorch:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = [256, 128, 64],
        activ: str = "relu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        early_stopping_metric: str = "val_mse",  # or "val_regret"
        grad_clip: Optional[float] = None,
        linear_impl: str = "manual",  # "manual" uses CustomLinear (custom backward), "autograd" uses nn.Linear
        device: str = "auto",
        save_dir: Optional[str] = None,
        random_seed: int = 42,
    ):
        _set_seed(random_seed)
        self.device = _select_device(device)
        self.model = _MLP(input_dim, output_dim, hidden_layers, activ, dropout, linear_impl).to(self.device)
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.grad_clip = grad_clip
        self.history = _History(train_loss=[], val_loss=[])
        self.save_dir = save_dir or os.path.join("runs", f"pt_mlp_{_timestamp()}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.output_dim = output_dim
        self.random_seed = random_seed

    # --------------
    # Public methods
    # --------------
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, verbose: bool = True):
        train_loader = DataLoader(NumpyDataset(X_train, Y_train), batch_size=self.batch_size, shuffle=True, drop_last=False)
        Xv = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        Yv = torch.from_numpy(Y_val.astype(np.float32)).to(self.device)

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_losses = []
            for xb_np, yb_np in train_loader:
                xb = xb_np.to(self.device)
                yb = yb_np.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                y_hat = self.model(xb)
                loss = self.criterion(y_hat, yb)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            # ---- validation ----
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xv)
                val_loss = float(self.criterion(val_pred, Yv).cpu())
                if self.early_stopping_metric == "val_regret":
                    preds = torch.argmin(val_pred, dim=1)
                    regret = torch.mean(Yv[torch.arange(Yv.size(0)), preds]).item()
                    monitor = regret
                else:
                    monitor = val_loss

            self.history.train_loss.append(float(np.mean(epoch_losses)))
            self.history.val_loss.append(monitor)

            if verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"[Epoch {epoch:03d}] train_loss={self.history.train_loss[-1]:.6f}  val_metric={monitor:.6f}")

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
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        Y_hat = self.model(X_t).cpu().numpy()
        return Y_hat

    @torch.no_grad()
    def predict_argmin(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        preds = torch.argmin(self.model(X_t), dim=1).cpu().numpy()
        return preds

    def plot_learning_curves(self):
        fig = plt.figure(figsize=(6,4))
        plt.plot(self.history.train_loss, label="train")
        plt.plot(self.history.val_loss, label="val")
        plt.xlabel("epoch")
        plt.ylabel("metric" if self.early_stopping_metric == "val_regret" else "MSE loss")
        plt.title("Learning curves")
        plt.legend()
        path = os.path.join(self.save_dir, "learning_curves.png")
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # --------------
    # Utilities
    # --------------
    def _save_config(self):
        cfg = dict(
            output_dim=self.output_dim,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_metric=self.early_stopping_metric,
            device=str(self.device),
            random_seed=self.random_seed,
        )
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)


# ---------------------------
# Metric: mean regret (helper)
# ---------------------------
def mean_test_regret(Y_true: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean([Y_true[j, preds[j]] for j in range(len(preds))]))


# ---------------------
# GNN building blocks
# ---------------------
class GraphConv(nn.Module):
    """Kipf-Welling style: H' = Â H W + b, with Â = D^{-1/2}(A+I)D^{-1/2}.
    Supports dense or sparse \u00c2.
    """
    def __init__(self, in_features: int, out_features: int, A_hat: torch.Tensor, bias: bool = True):
        super().__init__()
        self.A_hat = A_hat  # (N,N) dense or sparse_coo tensor on same device as inputs
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (N, in_features)
        A = self.A_hat
        if A.is_sparse:
            H_agg = torch.sparse.mm(A, H)
        else:
            H_agg = A @ H
        return self.lin(H_agg)


def normalize_adj(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    A = A.astype(np.float32)
    if add_self_loops:
        A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d + 1e-8, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ----------------
# Demo & smoke test
# ----------------
if __name__ == "__main__":
    # Random data sanity check
    N, D, K = 1024, 20, 8
    X = np.random.randn(N, D).astype(np.float32)
    Wtrue = np.random.randn(D, K).astype(np.float32)
    Y = X @ Wtrue + 0.1 * np.random.randn(N, K).astype(np.float32)

    # Split
    idx = np.arange(N)
    np.random.shuffle(idx)
    tr, va, te = idx[:700], idx[700:850], idx[850:]
    Xtr, Ytr = X[tr], Y[tr]
    Xva, Yva = X[va], Y[va]
    Xte, Yte = X[te], Y[te]

    model = MultiTargetMLPRegressorTorch(
        input_dim=D, output_dim=K, hidden_layers=[128,64], dropout=0.1,
        lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=200,
        early_stopping_patience=20, early_stopping_metric="val_mse",
        grad_clip=1.0, linear_impl="manual", device="auto", random_seed=42,
    )
    model.fit(Xtr, Ytr, Xva, Yva, verbose=True)

    Y_pred = model.predict(Xte)
    preds = model.predict_argmin(Xte)
    print("Mean test regret:", mean_test_regret(Yte, preds))
    print("Mean test regret per class:", Yte.mean(axis=0))

    print("Learning curves saved to:", model.plot_learning_curves())
