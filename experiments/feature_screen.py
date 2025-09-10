import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------- 设置统一保存目录 ---------
SAVE_DIR = Path("/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/out_dir")   # 统一修改这里
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --------- 1) Load data ---------
file_path = Path( SAVE_DIR / "analysis_ready_results_abs.csv")
df = pd.read_csv(file_path)

# --------- 2) Select features and target ---------
features = [
    "Iterations", "PrimalRes", "DualRes", "RelGap", 
    "ActiveLimits", "r_max", "t", "sum_r_sq", "sum_r_cu",
    "sep_max", "sep_mean", "sum_sep_sq", "tree_max_deg",
    "tree_h", "fillin", "coupling"
]
target = "SolveTime"

features_present = [c for c in features if c in df.columns]
corr_matrix = df[features_present + [target]].corr()

# --------- 3) Mask matrix so only SolveTime row/col keep values ---------
mask = corr_matrix.copy() * np.nan
idx = list(corr_matrix.columns).index(target)
mask.iloc[idx, :] = corr_matrix.iloc[idx, :]
mask.iloc[:, idx] = corr_matrix.iloc[:, idx]

# --------- 4) Plot heatmap ---------
plt.figure(figsize=(12, 9))
ax = sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0,
                 xticklabels=corr_matrix.columns,
                 yticklabels=corr_matrix.columns)

# Add black rectangles for SolveTime row/col
ax.add_patch(plt.Rectangle((idx, 0), 1, len(corr_matrix),
                           fill=False, edgecolor='black', lw=2))
ax.add_patch(plt.Rectangle((0, idx), len(corr_matrix), 1,
                           fill=False, edgecolor='black', lw=2))

# Overlay the correlation numbers only for SolveTime row/col
for i in range(len(corr_matrix)):
    if not np.isnan(mask.iloc[idx, i]):
        ax.text(i + 0.5, idx + 0.5, f"{mask.iloc[idx, i]:.2f}",
                ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    if not np.isnan(mask.iloc[i, idx]) and i != idx:
        ax.text(idx + 0.5, i + 0.5, f"{mask.iloc[i, idx]:.2f}",
                ha="center", va="center", color="black", fontsize=9, fontweight="bold")

plt.title("Correlation Heatmap (Features + SolveTime, Highlighted with Coeffs)")
plt.tight_layout()
out_path = SAVE_DIR / "heatmap_solvetime_with_coeffs.png"
plt.savefig(out_path, dpi=200)
plt.show()

print(f"✅ Saved figure: {out_path}")


selected_features = ["sum_r_cu", "sep_max", "fillin", "ActiveLimits", "tree_max_deg"]

plt.figure(figsize=(15, 10))
cases = df["Case"].unique()
colors = plt.cm.tab20.colors  # 最多 20 种颜色，够区分多个 case

for i, feat in enumerate(selected_features, 1):
    plt.subplot(2, 3, i)
    for j, case in enumerate(cases):
        subdf = df[df["Case"] == case]
        plt.scatter(subdf[feat], subdf["SolveTime"], 
                    color=colors[j % len(colors)], 
                    label=case if i == 1 else "",  # 只在第1幅子图显示图例
                    alpha=0.7, s=30)
    corr_val = df[[feat, "SolveTime"]].corr().iloc[0, 1]
    plt.xlabel(feat)
    plt.ylabel("SolveTime")
    plt.title(f"{feat} vs SolveTime\n(Pearson r={corr_val:.2f})")

# 只显示一次图例（放在第一个子图外）
plt.legend(title="Case", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(SAVE_DIR / "scatterplots_solvetime_vs_features_by_case.png", dpi=200)
plt.show()
