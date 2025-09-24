# ============================== supervised.py ==============================
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gcn_utils.logging import log
from trainers.loss import multitask_loss

# ---------------------------------------------------------------------------
# 你的原始训练/验证逻辑 —— 保持不变
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        if model.__class__.__name__ == "GCNGlobal":
            A, X, y_reg, y_cls, gvec = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
                batch["gvec"].to(device),
            )
            pred_yv, pred_y_reg = model(A, X, gvec)
        else:
            A, X, y_reg, y_cls = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
            )
            pred_yv, pred_y_reg = model(A, X)

        loss = loss_fn(pred_yv, pred_y_reg, y_reg, y_cls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * y_reg.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for batch in loader:
        if model.__class__.__name__ == "GCNGlobal":
            A, X, y_reg, y_cls, gvec = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
                batch["gvec"].to(device),
            )
            pred_yv, pred_y_reg = model(A, X, gvec)
        else:
            A, X, y_reg, y_cls = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
            )
            pred_yv, pred_y_reg = model(A, X)
        loss = loss_fn(pred_yv, pred_y_reg, y_reg, y_cls)
        total += loss.item() * y_reg.size(0)
    return total / len(loader.dataset)

def fit(model, train_ds, val_ds, epochs=50, batch_size=8, lr=3e-4,
        device="cpu", plot_path="./models/GNN/runners/loss_curve.png"):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=getattr(train_ds, "collate_fn", None))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=getattr(val_ds, "collate_fn", None))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = multitask_loss
    best = float("inf")

    train_losses, val_losses = [], []

    for ep in range(1, epochs+1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device)
        va = eval_epoch(model, val_loader, loss_fn, device)
        train_losses.append(tr)
        val_losses.append(va)
        log(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")
        best = min(best, va)

    # ---- 绘图 ----
    Path(os.path.dirname(plot_path)).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="train_loss")
    plt.plot(range(1, epochs+1), val_losses, label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    print(f"✅ Loss curve saved to {plot_path}")

    return {"val_mse": best}

# ---------------------------------------------------------------------------
# 数据集构建：同时支持
#  (A) 内部三分：samples → train/val/test（可固定内部 test 索引以便跨模型对比）
#  (B) 外部测试集：samples → train/val；external_test_samples → test
# ---------------------------------------------------------------------------

def _save_indices(indices: np.ndarray, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    if path.endswith(".npy"):
        np.save(path, indices)
    else:
        with open(path, "w") as f:
            json.dump({"indices": indices.tolist()}, f)

def _load_indices(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    else:
        with open(path, "r") as f:
            obj = json.load(f)
        return np.array(obj["indices"], dtype=np.int64)

def _make_internal_split_indices(
    n_total: int,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    fixed_internal_test_path: Optional[str] = None,
    create_if_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (train_idx, val_idx, test_idx) for INTERNAL split."""
    assert abs(sum(ratios) - 1.0) < 1e-8, "ratios must sum to 1.0"
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_total, dtype=np.int64)

    # 处理 test 索引（可固定）
    if fixed_internal_test_path and os.path.exists(fixed_internal_test_path):
        test_idx = _load_indices(fixed_internal_test_path)
        test_idx = test_idx[(test_idx >= 0) & (test_idx < n_total)]
        test_idx = np.unique(test_idx)
    else:
        n_test_target = int(round(ratios[2] * n_total))
        perm = rng.permutation(n_total)
        test_idx = np.sort(perm[:n_test_target])
        if fixed_internal_test_path and create_if_missing:
            _save_indices(test_idx, fixed_internal_test_path)

    # 从剩余中按比例切 train / val
    remain = np.setdiff1d(all_idx, test_idx, assume_unique=True)
    n_train_target = int(round(ratios[0] / (ratios[0] + ratios[1]) * len(remain)))
    perm_remain = rng.permutation(remain)
    train_idx = np.sort(perm_remain[:n_train_target])
    val_idx   = np.sort(perm_remain[n_train_target:])
    return train_idx, val_idx, np.sort(test_idx)

def build_datasets(
    *,
    samples: List[Dict[str, Any]],
    builder_fn,                 # e.g. builder.build
    node_feature_fn,            # e.g. pipeline.node_features
    dataset_class,              # e.g. OPFGraphDataset
    collate_fn_factory,         # e.g. make_collate_fn
    mode: str = "internal",     # "internal" or "external"
    ratios: Tuple[float,float,float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    fixed_internal_test_path: Optional[str] = None,
    external_test_samples: Optional[List[Dict[str, Any]]] = None,
):
    """
    构建 (train_ds, val_ds, test_ds, meta)

    mode="internal":
        - 从 samples 按 ratios 三分得到 train/val/test
        - 若 fixed_internal_test_path 提供，则固定内部 test 索引以利于跨模型比较
    mode="external":
        - 从 samples 仅切 train/val（按 ratios 的前两段比例）
        - test 来自 external_test_samples（与 samples 无关）
    """
    assert mode in ("internal", "external")

    if mode == "internal":
        n_total = len(samples)
        train_idx, val_idx, test_idx = _make_internal_split_indices(
            n_total=n_total,
            ratios=ratios,
            seed=seed,
            fixed_internal_test_path=fixed_internal_test_path,
            create_if_missing=True,
        )
        train_samples = [samples[i] for i in train_idx.tolist()]
        val_samples   = [samples[i] for i in val_idx.tolist()]
        test_samples  = [samples[i] for i in test_idx.tolist()]

    else:  # mode == "external"
        assert external_test_samples is not None and len(external_test_samples) > 0, \
            "external_test_samples must be provided for mode='external'."
        # 只在 samples 上切 train/val
        n_total = len(samples)
        # 用内部索引函数，但把 test 比例当作 0，只切 train/val 的相对比例
        r_train, r_val, _ = ratios
        rv_sum = r_train + r_val
        r_train_eff = r_train / rv_sum
        # 这里直接打乱并切分
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        n_train = int(round(r_train_eff * n_total))
        train_idx = np.sort(perm[:n_train])
        val_idx   = np.sort(perm[n_train:])
        train_samples = [samples[i] for i in train_idx.tolist()]
        val_samples   = [samples[i] for i in val_idx.tolist()]
        test_samples  = external_test_samples  # 外部测试集

    # 构造 Dataset
    train_ds = dataset_class(train_samples, builder_fn, node_feature_fn)
    val_ds   = dataset_class(val_samples,   builder_fn, node_feature_fn)
    test_ds  = dataset_class(test_samples,  builder_fn, node_feature_fn)

    # 维持你原有“挂接 collate_fn”的习惯
    train_ds.collate_fn = collate_fn_factory(train_ds)
    val_ds.collate_fn   = collate_fn_factory(val_ds)
    test_ds.collate_fn  = collate_fn_factory(test_ds)

    meta = {
        "mode": mode,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
    }
    if mode == "internal":
        meta["train_idx"] = train_idx
        meta["val_idx"] = val_idx
        meta["test_idx"] = test_idx
        meta["fixed_internal_test_path"] = fixed_internal_test_path

    return train_ds, val_ds, test_ds, meta

# ---------------------------------------------------------------------------
# 测试评估（分类+回归指标 + 作图 + 可选保存预测）
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_on_loader(model, loader, device):
    model.eval()
    all_logits, all_reg_pred, all_cls, all_reg_true = [], [], [], []
    for batch in loader:
        if model.__class__.__name__ == "GCNGlobal":
            A, X, y_reg, y_cls, gvec = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
                batch["gvec"].to(device),
            )
            logits, reg_pred = model(A, X, gvec)
        else:
            A, X, y_reg, y_cls = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
            )
            logits, reg_pred = model(A, X)
        all_logits.append(logits.detach().cpu())
        all_reg_pred.append(reg_pred.detach().cpu())
        all_cls.append(y_cls.detach().cpu())
        all_reg_true.append(y_reg.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_pred_cls = logits.argmax(dim=-1)
    y_true_cls = torch.cat(all_cls, dim=0)
    y_pred_reg = torch.cat(all_reg_pred, dim=0)     # [N, K]
    y_true_reg = torch.cat(all_reg_true, dim=0)     # [N, K]
    return logits, y_pred_cls, y_true_cls, y_pred_reg, y_true_reg

def _classification_metrics(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    acc = float((y_true == y_pred).mean())

    f1s = []
    for c in range(num_classes):
        tp = C[c, c]
        fp = C[:, c].sum() - tp
        fn = C[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion": C}

def _regression_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    assert yt.shape == yp.shape
    diff = yp - yt
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum((yt - yp)**2))
    ss_tot = float(np.sum((yt - yt.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    K = yt.shape[1] if yt.ndim == 2 else 1
    per_dim = []
    for k in range(K):
        yk = yt[:, k]; pk = yp[:, k]
        res = np.sum((yk - pk)**2)
        tot = np.sum((yk - yk.mean())**2)
        r2k = 1.0 - res / tot if tot > 0 else 0.0
        msek = np.mean((pk - yk)**2)
        maek = np.mean(np.abs(pk - yk))
        per_dim.append({"mse": float(msek), "mae": float(maek), "r2": float(r2k)})
    return {"mse": mse, "mae": mae, "r2": r2, "per_dim": per_dim}

def _plot_confusion_matrix(C, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    im = plt.imshow(C, interpolation='nearest')
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

def _plot_regression_parity(y_true, y_pred, out_path, max_dims_to_plot=6):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    K = yt.shape[1] if yt.ndim == 2 else 1
    kshow = min(K, max_dims_to_plot)
    cols = min(3, kshow); rows = int(np.ceil(kshow / cols))
    plt.figure(figsize=(4*cols, 3.5*rows))
    for k in range(kshow):
        ax = plt.subplot(rows, cols, k+1)
        ax.scatter(yt[:, k], yp[:, k], s=10, alpha=0.6)
        vmin = float(min(yt[:, k].min(), yp[:, k].min()))
        vmax = float(max(yt[:, k].max(), yp[:, k].max()))
        ax.plot([vmin, vmax], [vmin, vmax], linewidth=1)
        ax.set_xlabel(f"True y[{k}]"); ax.set_ylabel(f"Pred y[{k}]")
        ax.set_title(f"Parity dim {k}"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

def _plot_residual_hist(y_true, y_pred, out_path):
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    res = yp - yt
    plt.figure(figsize=(5, 3.8))
    plt.hist(res, bins=40, alpha=0.85)
    plt.xlabel("Residual (y_pred - y_true)"); plt.ylabel("Count")
    plt.title("Residual Histogram (All dims)"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

@torch.no_grad()
def test(model,
         test_ds,
         batch_size: int = 8,
         device: str = "cpu",
         out_dir: str = "./models/GNN/runners/test",
         loss_fn=multitask_loss,
         save_preds: bool = False):
    """
    在 test_ds 上评估（无论是内部三分 test，还是外部 test，都可直接使用）：
      - overall multitask loss（与 eval_epoch 口径一致）
      - 分类：accuracy, macro-F1, confusion（+图）
      - 回归：MSE/MAE/R2（overall & per-dim）（+parity & residual 图）
      - 可选保存预测：logits.npy, y_pred_reg.npy, y_true_cls.npy, y_true_reg.npy
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=getattr(test_ds, "collate_fn", None)
    )

    logits, y_pred_cls, y_true_cls, y_pred_reg, y_true_reg = _predict_on_loader(model, test_loader, device)

    # 与 eval_epoch 一致的 loss 口径
    total_loss = 0.0
    n_samples = len(test_loader.dataset)
    for batch in test_loader:
        if model.__class__.__name__ == "GCNGlobal":
            A, X, y_reg, y_cls, gvec = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
                batch["gvec"].to(device),
            )
            pred_yv, pred_y_reg = model(A, X, gvec)
        else:
            A, X, y_reg, y_cls = (
                batch["A_hat"].to(device),
                batch["X"].to(device),
                batch["y_reg"].to(device),
                batch["y_cls"].to(device),
            )
            pred_yv, pred_y_reg = model(A, X)
        loss = loss_fn(pred_yv, pred_y_reg, y_reg, y_cls)
        total_loss += loss.item() * y_reg.size(0)
    test_loss = total_loss / n_samples

    # 指标与作图
    cls_mets = _classification_metrics(y_true_cls.numpy(), y_pred_cls.numpy(),
                                       num_classes=logits.shape[-1])
    reg_mets = _regression_metrics(y_true_reg.numpy(), y_pred_reg.numpy())

    _plot_confusion_matrix(cls_mets["confusion"], os.path.join(out_dir, "confusion_matrix.png"))
    _plot_regression_parity(y_true_reg.numpy(), y_pred_reg.numpy(), os.path.join(out_dir, "reg_parity.png"))
    _plot_residual_hist(y_true_reg.numpy(), y_pred_reg.numpy(), os.path.join(out_dir, "residual_hist.png"))

    if save_preds:
        np.save(os.path.join(out_dir, "logits.npy"),        logits.numpy())
        np.save(os.path.join(out_dir, "y_pred_reg.npy"),    y_pred_reg.numpy())
        np.save(os.path.join(out_dir, "y_true_cls.npy"),    y_true_cls.numpy())
        np.save(os.path.join(out_dir, "y_true_reg.npy"),    y_true_reg.numpy())

    log(f"[TEST] loss {test_loss:.6f} | acc {cls_mets['accuracy']:.4f} | macro-F1 {cls_mets['macro_f1']:.4f} | "
        f"reg MSE {reg_mets['mse']:.6f} | MAE {reg_mets['mae']:.6f} | R2 {reg_mets['r2']:.4f}")
    print(f"📊 Saved test figures under: {out_dir}\n"
          f"   - confusion_matrix.png\n"
          f"   - reg_parity.png\n"
          f"   - residual_hist.png")

    return {
        "test_loss": test_loss,
        "classification": {
            "accuracy": cls_mets["accuracy"],
            "macro_f1": cls_mets["macro_f1"],
            "confusion": cls_mets["confusion"],
        },
        "regression": reg_mets,
    }

# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------
# 1) 内部三分（可固定内部 test 以跨模型对比）
#    train_ds, val_ds, test_ds, meta = build_datasets(
#        samples=samples,
#        builder_fn=builder.build,
#        node_feature_fn=pipeline.node_features,
#        dataset_class=OPFGraphDataset,
#        collate_fn_factory=make_collate_fn,
#        mode="internal",
#        ratios=(0.7, 0.15, 0.15),
#        seed=42,
#        fixed_internal_test_path="./models/GNN/splits/internal_test_indices.npy",
#    )
#
# 2) 外部测试集（跨模型对比常用）
#    train_ds, val_ds, test_ds, meta = build_datasets(
#        samples=samples,
#        builder_fn=builder.build,
#        node_feature_fn=pipeline.node_features,
#        dataset_class=OPFGraphDataset,
#        collate_fn_factory=make_collate_fn,
#        mode="external",
#        ratios=(0.8, 0.2, 0.0),  # 只用前两段比例切 train/val
#        seed=42,
#        external_test_samples=external_test_samples,  # 你单独加载
#    )
#
# 3) 训练与测试（两种模式用法一致）
#    fit_stats  = fit(model, train_ds, val_ds, epochs=50, batch_size=8, lr=3e-4, device=device)
#    test_stats = test(model, test_ds, batch_size=8, device=device,
#                      out_dir="./models/GNN/runners/test", save_preds=True)
# ===========================================================================
