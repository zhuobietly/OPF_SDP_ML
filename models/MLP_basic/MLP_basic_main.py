# demo_train_torch.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 可选：如果你已有这些绘图函数，会自动调用；没有就忽略
try:
    from utils.plot import plot_confusion_matrix, plot_regret_boxplot
except Exception:
    plot_confusion_matrix = None
    plot_regret_boxplot = None

from module import (
    MultiTargetMLPRegressorTorch,
    mean_test_regret,
)

# ==== 路径与配置 ====
DATA_DIR = "data/feature/XGboost/"
X_FILE = "X_PCA.csv"
Y_FILE = "Y_A.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==== 加载数据 ====
X = pd.read_csv(os.path.join(DATA_DIR, X_FILE)).values.astype(np.float32)
Y = pd.read_csv(os.path.join(DATA_DIR, Y_FILE)).values.astype(np.float32)
assert X.ndim == 2 and Y.ndim == 2, "Expect 2D arrays: X:(N,D), Y:(N,K)"

# ==== 切分 ====
idx = np.arange(X.shape[0])
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED)
X_train, Y_train = X[train_idx], Y[train_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=RANDOM_SEED
)

# ==== 模型 ====
model = MultiTargetMLPRegressorTorch(
    input_dim=X_train.shape[1],
    output_dim=Y_train.shape[1],
    hidden_layers=[256, 128, 64],
    dropout=0.10,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=200,
    early_stopping_patience=20,
    early_stopping_metric="val_mse",   # 或 "val_regret"
    linear_impl="manual",              # 手写 backward；或 "autograd"
    device="auto",
    random_seed=RANDOM_SEED,
)

# ==== 训练 ====
model.fit(X_train, Y_train, X_val, Y_val, verbose=True)

# ==== 预测与评估 ====
Y_pred = model.predict(X_test)             # (N,K)
preds  = model.predict_argmin(X_test)      # (N,)
mreg   = mean_test_regret(Y_test, preds)

print("Mean test regret:", mreg)
print("Mean test regret per class:", Y_test.mean(axis=0))

# 学习曲线
curve_path = model.plot_learning_curves()
print("Learning curves saved to:", curve_path)
print("save_dir:", model.save_dir)

# ==== 混淆矩阵（基于 argmin 选择的“类”） ====
y_true = np.argmin(Y_test, axis=1)
cm = confusion_matrix(y_true, preds)
print("Confusion matrix:\n", cm)

if plot_confusion_matrix is not None:
    plot_confusion_matrix(model.save_dir, y_true, preds)

# ==== 箱线图（每类 regret 分布，可选） ====
if plot_regret_boxplot is not None:
    Y_test_copy = plot_regret_boxplot(model.save_dir, Y_test, preds)
    print("Y_test_copy shape:", Y_test_copy.shape)
