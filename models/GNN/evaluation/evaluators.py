import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict
from registries import register
from .eval_utils import (
    _save_json, _plot_save, _save_metrics, _save_preds_targets,
    _logsumexp, _conf_mat, _cls_report
)

class BaseEvaluator(object):
    def __init__(self, output_dir="./outputs", device=None):
        self.output_dir = output_dir
        self.device = device
        os.makedirs(self.output_dir, exist_ok=True)
        self._reset()

    def _reset(self): ...
    def update(self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        self._accumulate(batch, model_out)
    def _accumulate(self, batch, model_out): ...
    def compute(self):
        return self._finalize(self.output_dir)
    @torch.no_grad()
    def run(self, model, loader, device=None, output_dir=None):
        """标准测试入口：清空缓存 → 遍历 loader 前向 → 累积 → 汇总"""
        self._reset()
        output_dir = output_dir or self.output_dir
        dev = device or self.device
        model.eval()
        for batch in loader:
            # 搬到 device（递归支持 dict）
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(dev)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(dev)
            out = model(batch)
            self._accumulate(batch, out)
        return self._finalize(output_dir)


@register("evaluator", "arr_cls")
class ArrClsEvaluator(BaseEvaluator):
    """
    数组回归 + 分类：
      * 不使用 mask
      * 分类由回归数组派生：
          logits = -(y_arr_reg_pred - center) / tau
      * 计算 MAE / RMSE / 分类准确率 / 可选交叉熵
    """
    def __init__(self, output_dir, device=None, *,
                 tau: float = 1.0, center: bool = True,
                 report_ce: bool = True, save_plots: bool = True):
        super().__init__(output_dir, device)
        self.tau = float(tau)
        self.center = bool(center)
        self.report_ce = bool(report_ce)
        self.save_plots = bool(save_plots)

    def _reset(self):
        self._arr_p: List[torch.Tensor] = []
        self._arr_t: List[torch.Tensor] = []
        self._ytrue: List[torch.Tensor] = []

    def _accumulate(self, batch, model_out):
        P = model_out["pred_arr_reg"].detach().cpu()
        T = batch["y_arr_reg"].detach().cpu()
        self._arr_p.append(P)
        self._arr_t.append(T)
        if "y_cls" in batch:
            self._ytrue.append(batch["y_cls"].detach().cpu())

    def _finalize(self, out_dir: str):
        P = torch.cat(self._arr_p, dim=0).numpy()  # [N,K]
        T = torch.cat(self._arr_t, dim=0).numpy()
        N, K = P.shape

        # --- 数组回归 ---
        diff = P - T
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        metrics = {"array": {"MAE": mae, "RMSE": rmse}, "classify": {}}

        # --- 分类 logits ---
        if self.center:
            P_center = P - P.mean(axis=1, keepdims=True)
            Z = -P_center / self.tau
        else:
            Z = -P / self.tau

        yhat = Z.argmax(axis=1)
        Ytrue = torch.cat(self._ytrue, dim=0).numpy() if self._ytrue else T.argmin(axis=1)
        acc = float((yhat == Ytrue).mean())
        metrics["classify"]["acc"] = acc

        # --- 可选 CE ---
        if self.report_ce and len(Ytrue) == N:
            ce = float(np.mean(-Z[np.arange(N), Ytrue] + _logsumexp(Z, axis=1)))
            metrics["classify"]["cross_entropy"] = ce

        # --- 绘图 ---
        if self.save_plots:
            # 误差直方图
            fig = plt.figure()
            plt.hist(diff.reshape(-1), bins=40)
            plt.xlabel("Element-wise Error (pred - true)")
            plt.ylabel("Count")
            plt.title("Array Regression Error")
            _plot_save(fig, os.path.join(out_dir, "arr_error_hist.png"))

            # every dimension MAE
            per_dim_mae = np.mean(np.abs(diff), axis=0)
            fig = plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(per_dim_mae)), per_dim_mae)
            plt.xlabel("Dimension")
            plt.ylabel("MAE")
            plt.title("Per-dimension MAE")
            plt.grid(True, alpha=0.3)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "arr_per_dim_mae.png"))
            
            
            #every dimension RMSE
            per_dim_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
            fig = plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(per_dim_rmse)), per_dim_rmse)
            plt.xlabel("Dimension")
            plt.ylabel("RMSE")
            plt.title("Per-dimension RMSE")
            plt.grid(True, alpha=0.3)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "arr_per_dim_rmse.png"))
            
            
            # 混淆矩阵
            nc = 15
            cm = _conf_mat(Ytrue, yhat, nc)
            fig = plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            _plot_save(fig, os.path.join(out_dir, "cls_confusion_matrix.png"))
            metrics["classify"].update(_cls_report(cm))

        # --- 保存结果 ---
        _save_metrics(metrics, out_dir)
        _save_preds_targets(P, T, out_dir)
        return metrics
