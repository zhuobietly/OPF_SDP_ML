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


# ====================== 新增：标量回归评估图辅助函数 ======================
def _plot_scalar_parity(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path,
                        title: str = "Scalar Regression: y_pred vs y_true") -> None:
    """Parity plot: y_true on x-axis, y_pred on y-axis, with y=x reference."""
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = plt.gca()
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    # y = x 参考线
    vmin = min(np.min(y_true), np.min(y_pred))
    vmax = max(np.max(y_true), np.max(y_pred))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--")
    ax.set_xlabel("True scalar (y)")
    ax.set_ylabel("Predicted scalar (ŷ)")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scalar_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    """Residual diagnostics: histogram of residuals and residual vs prediction."""
    residuals = y_pred - y_true

    # 残差直方图
    fig1 = plt.figure(figsize=(6.5, 5.0))
    ax1 = plt.gca()
    ax1.hist(residuals, bins=40)
    ax1.set_title("Scalar Regression: Residuals Histogram (ŷ - y)")
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Count")
    fig1.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_dir / "scalar_reg_residual_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 残差 vs 预测
    fig2 = plt.figure(figsize=(6.5, 5.0))
    ax2 = plt.gca()
    ax2.scatter(y_pred, residuals, s=10, alpha=0.6)
    ax2.axhline(0.0, linestyle="--")
    ax2.set_title("Scalar Regression: Residuals vs Prediction")
    ax2.set_xlabel("Predicted scalar (ŷ)")
    ax2.set_ylabel("Residual (ŷ - y)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "scalar_reg_residual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
# ======================================================================


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
        pred_label = pred_arr_reg.argmin(dim=1)

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
    num_classes = 15
    cm = _confusion_matrix(y_true_np, y_pred_np, num_classes)

    # 数组回归
    mae_arr  = sum_abs_arr / (n * K_arr)
    rmse_arr = math.sqrt(sum_sq_arr / (n * K_arr))
    per_mae_arr  = (per_dim_abs / n).cpu().tolist()
    per_rmse_arr = (torch.sqrt(per_dim_sq / n)).cpu().tolist()

    # 标量回归
    mae_scalar  = sum_abs_scalar / n
    rmse_scalar = math.sqrt(sum_sq_scalar / n)

    # === 新增：R^2 ===
    y_true_scalar_np_full = np.concatenate(all_true_scalar).reshape(-1)
    y_pred_scalar_np_full = np.concatenate(all_pred_scalar).reshape(-1)
    ss_res = float(np.sum((y_true_scalar_np_full - y_pred_scalar_np_full) ** 2))
    y_mean = float(np.mean(y_true_scalar_np_full)) if y_true_scalar_np_full.size else 0.0
    ss_tot = float(np.sum((y_true_scalar_np_full - y_mean) ** 2))
    r2_scalar = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

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
        "scalar_reg_r2": r2_scalar,  # 新增
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
        # 图：混淆矩阵（保留你的）
        _plot_confusion_matrix(cm, class_names, "Confusion Matrix", p / "confusion_matrix.png", False)
        _plot_confusion_matrix(cm, class_names, "Confusion Matrix (norm)", p / "confusion_matrix_norm.png", True)
        # 图：标量回归（新增的三张）
        _plot_scalar_parity(
            y_true_scalar_np_full,
            y_pred_scalar_np_full,
            p / "scalar_reg_parity.png",
            title=f"Scalar Regression Parity (R²={r2_scalar:.4f}, MAE={mae_scalar:.4f}, RMSE={rmse_scalar:.4f})"
        )
        _plot_scalar_residuals(y_true_scalar_np_full, y_pred_scalar_np_full, p)

    return metrics
