# trainers/evaluate.py
import math
import json
from pathlib import Path
from typing import Optional, Dict, Any, Sequence

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix without sklearn: shape [C, C], rows=true, cols=pred."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _plot_confusion_matrix(cm: np.ndarray,
                           class_names: Optional[Sequence[str]],
                           title: str,
                           out_path: Path,
                           normalize: bool = False) -> None:
    """Plot and save confusion matrix using matplotlib."""
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_disp = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
    else:
        cm_disp = cm

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = plt.gca()
    im = ax.imshow(cm_disp, interpolation='nearest')
    ax.set_title(title)

    num_classes = cm.shape[0]
    tick_labels = class_names if class_names is not None else [str(i) for i in range(num_classes)]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    # Annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm_disp.max() / 2.0 if cm_disp.size else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_disp[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color="white" if val > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# @torch.no_grad()
# def evaluate(
#     model,
#     dataset,
#     batch_size: int = 8,
#     device: str = "cpu",
#     save_dir: Optional[str] = None,
#     class_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     """模型永远返回 (pred_arr_reg, pred_y_reg, pred_y_cls)
#        数据永远含 y_cls, y_arr_reg, y_reg
#     """
#     model.eval().to(device)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         collate_fn=getattr(dataset, "collate_fn", None))

#     total = 0
#     top1  = 0
#     sum_abs = 0.0
#     sum_sq  = 0.0
#     per_dim_abs = None
#     per_dim_sq  = None

#     all_pred_cls, all_true_cls = [], []
#     all_pred_reg, all_true_reg = [], []

#     for batch in loader:
#         A   = batch["A_hat"].to(device)
#         X   = batch["X"].to(device)
#         gvec = batch.get("gvec")
#         if gvec is not None:
#             gvec = gvec.to(device)

#         # 真值
#         y_cls = batch["y_cls"].to(device).long()
#         y_arr_reg = batch["y_arr_reg"].to(device).float()   # [B, K]
#         y_reg = batch["y_reg"].to(device).float()           # [B, K]
#         # 前向
#         pred_arr_reg, pred_y_reg, pred_y_cls = (
#             model(A, X, gvec) if gvec is not None else model(A, X)
#         )

#         # ---------- 分类 ----------
#         pred_label = pred_y_cls
#         top1 += (pred_label == y_cls).sum().item()

#         # ---------- 回归 ----------
#         diff = pred_arr_reg - y_arr_reg          # [B, K]
#         sum_abs += diff.abs().sum().item()
#         sum_sq  += (diff ** 2).sum().item()

#                 # ---------- 回归 ----------
#         diff = pred_arr_reg - y_arr_reg          # [B, K]
#         sum_abs += diff.abs().sum().item()
#         sum_sq  += (diff ** 2).sum().item()
#         if per_dim_abs is None:
#             K = diff.shape[1]
#             per_dim_abs = torch.zeros(K, device=device)
#             per_dim_sq  = torch.zeros(K, device=device)
#         per_dim_abs += diff.abs().sum(0)
#         per_dim_sq  += (diff ** 2).sum(0)

#         # 存档
#         all_pred_cls.append(pred_label.cpu().numpy())
#         all_true_cls.append(y_cls.cpu().numpy())
#         all_pred_reg.append(pred_arr_reg.detach().cpu().numpy())
#         all_true_reg.append(y_reg.cpu().numpy())

#         total += y_cls.size(0)

#     # ----- 汇总 -----
#     acc = top1 / total
#     y_pred_np = np.concatenate(all_pred_cls)
#     y_true_np = np.concatenate(all_true_cls)
#     num_classes = int(max(y_true_np.max(), y_pred_np.max())) + 1
#     cm = _confusion_matrix(y_true_np, y_pred_np, num_classes)

#     mae  = sum_abs / (total * K)
#     rmse = math.sqrt(sum_sq / (total * K))
#     per_mae  = (per_dim_abs / total).cpu().tolist()
#     per_rmse = (torch.sqrt(per_dim_sq / total)).cpu().tolist()

#     metrics = {
#         "num_samples"      : total,
#         "num_outputs"      : K,
#         "cls_accuracy_top1": acc,
#         "confusion_matrix" : cm.tolist(),
#         "reg_overall_mae"  : mae,
#         "reg_overall_rmse" : rmse,
#         "reg_per_dim_mae"  : per_mae,
#         "reg_per_dim_rmse" : per_rmse,
#     }

#     # ----- 保存 -----
#     if save_dir:
#         p = Path(save_dir)
#         p.mkdir(parents=True, exist_ok=True)
#         np.save(p / "pred_cls.npy", y_pred_np)
#         np.save(p / "true_cls.npy", y_true_np)
#         np.save(p / "pred_reg.npy", np.concatenate(all_pred_reg))
#         np.save(p / "true_reg.npy", np.concatenate(all_true_reg))
#         with open(p / "metrics.json", "w") as f:
#             json.dump(metrics, f, indent=2)
#         _plot_confusion_matrix(cm, class_names, "Confusion Matrix", p / "confusion_matrix.png", False)
#         _plot_confusion_matrix(cm, class_names, "Confusion Matrix (norm)", p / "confusion_matrix_norm.png", True)

#     return metrics
@torch.no_grad()
def evaluate(
    model,
    dataset,
    batch_size: int = 8,
    device: str = "cpu",
    save_dir: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    模型永远返回 (pred_arr_reg, pred_y_reg, pred_y_cls)
    数据永远有 (y_arr_reg, y_reg, y_cls)
    六个全部一起比
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=getattr(dataset, "collate_fn", None))

    n = 0
    # ---------- 分类 ----------
    top1 = 0
    all_pred_cls, all_true_cls = [], []
    # ---------- 数组回归 ----------
    sum_abs_arr = 0.0
    sum_sq_arr  = 0.0
    per_dim_abs = per_dim_sq = None
    K_arr = None
    all_pred_arr, all_true_arr = [], []
    # ---------- 标量回归 ----------
    sum_abs_scalar = 0.0
    sum_sq_scalar  = 0.0
    all_pred_scalar, all_true_scalar = [], []

    for batch in loader:
        A    = batch["A_hat"].to(device)
        X    = batch["X"].to(device)
        gvec = batch.get("gvec")
        if gvec is not None:
            gvec = gvec.to(device)

        # 真值
        y_cls    = batch["y_cls"].to(device).long()
        y_arr    = batch["y_arr_reg"].to(device).float()
        y_scalar = batch["y_reg"].to(device).float().squeeze()

        # 前向
        pred_arr_reg, pred_y_reg, pred_y_cls = (
            model(A, X, gvec) if gvec is not None else model(A, X)
        )
        arr_cls = -pred_arr_reg
        pred_label = arr_cls.argmax(dim=1)

        B = y_cls.size(0)
        n += B

        # 1. 分类
        top1 += (pred_label == y_cls).sum().item()
        all_pred_cls.append(pred_label.cpu().numpy())
        all_true_cls.append(y_cls.cpu().numpy())

        # 2. 数组回归
        diff_arr = pred_arr_reg - y_arr
        sum_abs_arr += diff_arr.abs().sum().item()
        sum_sq_arr  += (diff_arr ** 2).sum().item()
        if per_dim_abs is None:
            K_arr = y_arr.shape[1]
            per_dim_abs = torch.zeros(K_arr, device=device)
            per_dim_sq  = torch.zeros(K_arr, device=device)
        per_dim_abs += diff_arr.abs().sum(0)
        per_dim_sq  += (diff_arr ** 2).sum(0)
        all_pred_arr.append(pred_arr_reg.detach().cpu().numpy())
        all_true_arr.append(y_arr.cpu().numpy())

        # 3. 标量回归
        diff_scalar = pred_y_reg.squeeze() - y_scalar
        sum_abs_scalar += diff_scalar.abs().sum().item()
        sum_sq_scalar  += (diff_scalar ** 2).sum().item()
        all_pred_scalar.append(pred_y_reg.squeeze().detach().cpu().numpy())
        all_true_scalar.append(y_scalar.cpu().numpy())

    # ----- 汇总 -----
    # 分类
    acc = top1 / n
    y_pred_np = np.concatenate(all_pred_cls)
    y_true_np = np.concatenate(all_true_cls)
    num_classes = 9
    cm = _confusion_matrix(y_true_np, y_pred_np, num_classes)

    # 数组回归
    mae_arr  = sum_abs_arr / (n * K_arr)
    rmse_arr = math.sqrt(sum_sq_arr / (n * K_arr))
    per_mae_arr  = (per_dim_abs / n).cpu().tolist()
    per_rmse_arr = (torch.sqrt(per_dim_sq / n)).cpu().tolist()

    # 标量回归
    mae_scalar  = sum_abs_scalar / n
    rmse_scalar = math.sqrt(sum_sq_scalar / n)

    metrics = {
        "num_samples": n,
        "cls_accuracy_top1": acc,
        "confusion_matrix": cm.tolist(),
        "arr_reg_overall_mae": mae_arr,
        "arr_reg_overall_rmse": rmse_arr,
        "arr_reg_per_dim_mae": per_mae_arr,
        "arr_reg_per_dim_rmse": per_rmse_arr,
        "scalar_reg_mae": mae_scalar,
        "scalar_reg_rmse": rmse_scalar,
    }

    # ----- 保存 -----
    if save_dir:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        # 分类
        np.save(p / "pred_cls.npy", y_pred_np)
        np.save(p / "true_cls.npy", y_true_np)
        # 数组回归
        np.save(p / "pred_arr_reg.npy", np.concatenate(all_pred_arr))
        np.save(p / "true_arr_reg.npy", np.concatenate(all_true_arr))
        # 标量回归
        np.save(p / "pred_scalar_reg.npy", np.concatenate(all_pred_scalar))
        np.save(p / "true_scalar_reg.npy", np.concatenate(all_true_scalar))
        # 指标
        with open(p / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # 图
        _plot_confusion_matrix(cm, class_names, "Confusion Matrix", p / "confusion_matrix.png", False)
        _plot_confusion_matrix(cm, class_names, "Confusion Matrix (norm)", p / "confusion_matrix_norm.png", True)

    return metrics