# train_gcn_from_pkls.py
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from module import GCNBasicRegressorTorch, GraphSamplesDataset

# ===== 1) 读取你已经保存的两个 PKL =====
FEAT_PKL = "/home/goatoine/Documents/Lanyue/data/feature/GNN/gnn_X.pkl"   # 里面应至少有两列：'gnn_A_hat' 和 'gnn_X'（object，每格是 ndarray）
Y_PKL    = "/home/goatoine/Documents/Lanyue/data/feature/GNN/gnn_Y.pkl"       # 可以是 DataFrame(含 Chordal_* 列) 或 直接是 ndarray 的 pkl

feat = pd.read_pickle(FEAT_PKL)
if not isinstance(feat, pd.DataFrame):
    raise TypeError("gnn_X.pkl 应该是 DataFrame（至少包含 'gnn_A_hat' 和 'gnn_X' 两列）。")
need_cols = {"gnn_A_hat", "gnn_X"}
if not need_cols.issubset(feat.columns):
    raise KeyError(f"缺少列：{need_cols - set(feat.columns)}")

# 取成 Python 列表（不复制数据，便于 Dataset 按位置索引）
A_list = feat["gnn_A_hat"].to_list()
X_list = feat["gnn_X"].to_list()

# 读 Y
Y_loaded = pd.read_pickle(Y_PKL)
if isinstance(Y_loaded, pd.DataFrame):
    # 只要包含 'chordal' 的列（不区分大小写）
    y_cols = [c for c in Y_loaded.columns if "chordal" in c.lower()]
    if not y_cols:
        raise ValueError("Y.pkl 里没找到包含 'chordal' 的列。")
    # 可选：按后缀数字排序，保证列顺序稳定
    def _col_key(c):
        m = re.search(r'(\d+(?:\.\d+)?)$', c)
        return (re.sub(r'_\d+(?:\.\d+)?$', '', c), float(m.group(1)) if m else float('inf'))
    y_cols = sorted(y_cols, key=_col_key)
    Y = Y_loaded[y_cols].to_numpy(dtype=np.float32)
else:
    # 如果你把 ndarray 直接 to_pickle 了，也兼容
    Y = np.asarray(Y_loaded, dtype=np.float32)

# 一些健壮性检查
n = len(A_list)
assert len(X_list) == n and len(Y) == n, f"样本数不一致：A={len(A_list)}, X={len(X_list)}, Y={len(Y)}"
N, F = X_list[0].shape
K = Y.shape[1] if Y.ndim == 2 else 1

print(f"Loaded: n={n}, N={N}, F={F}, K={K}")

# ===== 2) 划分训练/验证 =====
idx_tr, idx_va = train_test_split(np.arange(n), test_size=0.2, random_state=42, shuffle=True)
tr_set = GraphSamplesDataset([A_list[i] for i in idx_tr], [X_list[i] for i in idx_tr], Y[idx_tr])
va_set = GraphSamplesDataset([A_list[i] for i in idx_va], [X_list[i] for i in idx_va], Y[idx_va])

# ===== 3) 建模训练 =====
reg = GCNBasicRegressorTorch(
    in_features=F, hidden=[64, 64], out_dim=K,
    lr=1e-3, weight_decay=1e-4, batch_size=32,
    max_epochs=200, early_stopping_patience=20,
    early_stopping_metric="val_mse",   # 也可以 "val_regret"
    device="auto", random_seed=42
)
reg.fit(tr_set, va_set, verbose=True)

# ===== 4) 预测与最优方案索引 =====
Y_val_pred = reg.predict(va_set)            # (num_val, K)
preds      = reg.predict_argmin(va_set)     # (num_val,)
print("Pred :", Y_val_pred, "Argmin preds:", preds)

# ===== 5) 保存学习曲线图 =====
curve_path = reg.plot_learning_curves()
print("Learning curves saved to:", curve_path)
