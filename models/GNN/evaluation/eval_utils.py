# models/GNN/trainers/evaluators/utils_eval.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "_save_json", "_plot_save", "_conf_mat",
    "_cls_report", "_save_metrics", "_save_preds_targets",
    "_logsumexp"
]

def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _save_metrics(metrics, out_dir):
    _save_json(metrics, os.path.join(out_dir, "metrics.json"))

def _save_preds_targets(preds, targets, out_dir):
    np.save(os.path.join(out_dir, "preds.npy"), preds)
    np.save(os.path.join(out_dir, "targets.npy"), targets)

def _plot_save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def _logsumexp(z, axis=1):
    """数值稳定版 logsumexp"""
    zmax = np.max(z, axis=axis, keepdims=True)
    return (zmax + np.log(np.sum(np.exp(z - zmax), axis=axis, keepdims=True))).squeeze(axis)

def _conf_mat(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def _cls_report(cm):
    """生成 per-class precision/recall/F1 及 overall accuracy"""
    nc = cm.shape[0]
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0) + 1e-12)
    rec  = np.divide(tp, cm.sum(1) + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    acc  = tp.sum() / np.maximum(cm.sum(), 1)
    return {
        "accuracy": float(acc),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1.tolist(),
    }
