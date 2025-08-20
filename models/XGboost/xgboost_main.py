import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import gc
# import torch  # 你如果用 PyTorch 再打开
from MXGB_ES import MultiTargetXGBRegressorWithEarlyStop
from utils.plot import *

# ==== 路径与配置 ====
DATA_DIR = "data/feature/XGboost/"
X_FILE = "new_X_PCA.csv"
Y_FILE = "new_Y_A.csv"
RANDOM_SEED = 42

# ==== 加载数据 ====
X = pd.read_csv(os.path.join(DATA_DIR, X_FILE)).values
Y = pd.read_csv(os.path.join(DATA_DIR, Y_FILE)).values

idx = np.arange(X.shape[0])
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED)
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=RANDOM_SEED)

# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
gc.collect()


# ==== 模型训练 ====
model = MultiTargetXGBRegressorWithEarlyStop()
model.fit(X_train, Y_train, X_val, Y_val, verbose=True)

# ==== 预测与评估 ====
Y_pred = model.predict(X_test)
preds = model.predict_argmin(X_test)
mean_regret = np.mean([Y_test[j, preds[j]] for j in range(len(preds))])

print("Mean test regret:", mean_regret)
print("Mean test regret per class:", Y_test.mean(axis=0))

# ==== 画学习曲线 ====
model.plot_learning_curves()

# ==== 混淆矩阵 ====
# 混淆矩阵
plot_confusion_matrix(model.save_dir, np.argmin(Y_test, axis=1), preds)


# ==== 箱线图 ====
# 箱线图
Y_test_copy = plot_regret_boxplot(model.save_dir, Y_test, preds)
print("Y_test_copy shape:", Y_test_copy.shape)


