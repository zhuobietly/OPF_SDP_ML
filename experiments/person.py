# -*- coding: utf-8 -*-
"""
只显示 SolveTime 行的相关系数：
- corr_row_CASE.csv：SolveTime 与各特征的皮尔逊相关系数（单行）
- heatmap_solvetime_row_CASE.png：单行热力图，|ρ|≥0.40 的数值加粗

依赖：pandas / numpy / matplotlib / seaborn
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 路径 ----------
SAVE_DIR = Path("/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/out_dir")
CSV_PATH = SAVE_DIR / "analysis_ready_results_abs.csv"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 读数据 ----------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"找不到数据：{CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# ---------- 配置 ----------
target = "SolveTime"
# 只要这些列里“存在于数据”的会被使用（容错）
candidate_features = [
    "Iterations", "PrimalRes", "DualRes", "RelGap",
    "r_max", "t", "sum_r_sq", "sum_r_cu",
    "sep_max", "sep_mean", "sum_sep_sq",
    "tree_max_deg", "tree_h", "fillin", "coupling"
]
cases_to_run = ["case2746wop"]   # 需要的话可改为多个 case
BOLD_THRESH = 0.40               # |ρ| >= 0.40 加粗

# ---------- 主流程 ----------
if "Case" not in df.columns or target not in df.columns:
    raise ValueError("数据缺少 'Case' 或 'SolveTime' 列。")

for case in cases_to_run:
    sub = df[df["Case"].astype(str) == str(case)].copy()
    if sub.empty:
        print(f"⚠️ 未找到 {case}，跳过。")
        continue

    # 选取实际存在的列
    features_present = [c for c in candidate_features if c in sub.columns]
    cols = features_present + [target]
    if len(features_present) == 0:
        print(f"⚠️ {case} 中没有找到任何候选特征列，跳过。")
        continue

    # 计算相关矩阵并取 SolveTime 行
    corr = sub[cols].corr(method="pearson")
    if target not in corr.index:
        print(f"⚠️ {case} 相关矩阵中没有 {target}，跳过。")
        continue
    row = corr.loc[target, cols]   # 一个 Series（顺序与 cols 一致）

    # --- 保存单行 CSV ---
    out_csv = SAVE_DIR / f"corr_row_{case}.csv"
    row.to_frame(name=target).T.to_csv(out_csv, index=True)
    print(f"✅ 保存：{out_csv}")

    # --- 只画单行热力图 ---
    # 把单行 Series 变成 1xN 的 DataFrame，行名就是 SolveTime
    row_df = pd.DataFrame([row.values], columns=cols, index=[target])

    # 画布宽度随列数调整
    fig_w = max(5, 0.6 * len(cols))
    fig_h = 2
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        row_df,
        annot=False,              # 数值我们手动写，以便控制加粗
        cmap="coolwarm",
        vmin=-1, vmax=1, center=0,
        linewidths=0.5, linecolor="white",
        cbar=True
    )

    # 坐标与标题
    ax.set_yticklabels([target], rotation=0)
    ax.set_xticklabels(cols, rotation=60, ha="right")
    ax.set_title(f"SolveTime correlations — {case}", pad=10)

    # 在每个格子居中标数值；|ρ|≥BOLD_THRESH 的加粗
    vals = row_df.values[0]
    for j, v in enumerate(vals):
        txt = f"{v:.2f}" if np.isfinite(v) else "NaN"
        ax.text(j + 0.5, 0.5, txt,
                ha="center", va="center",
                color="black",
                fontsize=9,
                fontweight=("bold" if (np.isfinite(v) and abs(v) >= BOLD_THRESH) else "normal"))

    # 给整行加一个黑色边框更醒目（单行时其实可有可无）
    ax.add_patch(plt.Rectangle((0, 0), len(cols), 1,
                               fill=False, edgecolor="black", lw=2))

    plt.tight_layout()
    out_png = SAVE_DIR / f"heatmap_solvetime_row_{case}.png"
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"✅ 保存：{out_png}")

print("🎉 完成。")
