# -*- coding: utf-8 -*-
"""
按 case 分别计算相关性矩阵（case1888, case1951），并输出：
1) corr_matrix_CASE.csv（完整皮尔逊相关系数矩阵）
2) heatmap_CASE.png（完整热力图）
3) heatmap_solvetime_only_CASE.png（只强调 SolveTime 行/列并标注数值）

运行环境：
- 需要 pandas / numpy / matplotlib / seaborn
- CSV 数据默认路径：/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/out_dir/analysis_ready_results_abs.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------- 0) 统一保存目录（与现有脚本保持一致）---------
SAVE_DIR = Path("/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/out_dir")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --------- 1) 加载数据 ---------
file_path = SAVE_DIR / "analysis_ready_results_abs.csv"
if not file_path.exists():
    raise FileNotFoundError(f"找不到数据文件：{file_path}")

df = pd.read_csv(file_path)

# 必要字段检查
required_cols = {"Case", "SolveTime"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"数据缺少必要列：{missing}，请检查 CSV")

# --------- 2) 选择特征与目标（按你之前脚本的列名）---------
features = [
    "Iterations", "PrimalRes", "DualRes", "RelGap",
    "ActiveLimits", "r_max", "t", "sum_r_sq", "sum_r_cu",
    "sep_max", "sep_mean", "sum_sep_sq", "tree_max_deg",
    "tree_h", "fillin", "coupling"
]
target = "SolveTime"

# 实际存在的列（容错）
features_present = [c for c in features if c in df.columns]
if not features_present:
    raise ValueError("在数据中没有找到任何特征列，请检查列名是否和脚本一致。")
cols_for_corr = features_present + [target]

# --------- 3) 仅处理指定的两个 case ----------
cases_to_run = ["case1888rte", "case1951rte","case2746wop"]

# 实用函数：画完整热力图
def plot_full_heatmap(corr_matrix: pd.DataFrame, title: str, out_path: Path):
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

# 实用函数：只强调 SolveTime 行/列并标注数值
def plot_solvetime_heatmap(corr_matrix: pd.DataFrame, target: str, title: str, out_path: Path):
    if target not in corr_matrix.columns:
        raise ValueError(f"目标列 {target} 不在相关性矩阵中。")

    idx = list(corr_matrix.columns).index(target)

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(
        corr_matrix, annot=False,
        cmap="coolwarm", center=0,
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns
    )
    # 用黑框强调目标行/列
    ax.add_patch(plt.Rectangle((idx, 0), 1, len(corr_matrix),
                               fill=False, edgecolor='black', lw=2))
    ax.add_patch(plt.Rectangle((0, idx), len(corr_matrix), 1,
                               fill=False, edgecolor='black', lw=2))

    # 在目标行/列处叠加数值标注
    for i in range(len(corr_matrix)):
        # 行：SolveTime 对其他列
        ax.text(i + 0.5, idx + 0.5,
                f"{corr_matrix.iloc[idx, i]:.2f}",
                ha="center", va="center", color="black",
                fontsize=9, fontweight="bold")
        # 列：其他列对 SolveTime（避免重复在对角）
        if i != idx:
            ax.text(idx + 0.5, i + 0.5,
                    f"{corr_matrix.iloc[i, idx]:.2f}",
                    ha="center", va="center", color="black",
                    fontsize=9, fontweight="bold")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

# --------- 4) 主流程：逐 case 计算并保存 ---------
for case in cases_to_run:
    subdf = df[df["Case"] == case].copy()
    if subdf.empty:
        print(f"⚠️ 警告：在数据中没有找到 {case}，跳过。")
        continue

    # 只取存在的列参与相关性计算（皮尔逊）
    corr_matrix = subdf[cols_for_corr].corr(method="pearson")

    # 保存 CSV
    csv_out = SAVE_DIR / f"corr_matrix_{case}.csv"
    corr_matrix.to_csv(csv_out, index=True)
    print(f"✅ 已保存相关性矩阵：{csv_out}")

    # 完整热力图
    fig_out_full = SAVE_DIR / f"heatmap_{case}.png"
    plot_full_heatmap(
        corr_matrix,
        title=f"Correlation Heatmap ({case})",
        out_path=fig_out_full
    )
    print(f"✅ 已保存完整热力图：{fig_out_full}")

    # 仅强调 SolveTime 的热力图
    fig_out_solvetime = SAVE_DIR / f"heatmap_solvetime_only_{case}.png"
    plot_solvetime_heatmap(
        corr_matrix, target=target,
        title=f"SolveTime-focused Correlations ({case})",
        out_path=fig_out_solvetime
    )
    print(f"✅ 已保存 SolveTime 强调热力图：{fig_out_solvetime}")

print("🎉 全部完成。")
