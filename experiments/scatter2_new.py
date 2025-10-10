# -*- coding: utf-8 -*-
# Scatter plots of SolveTime vs selected features (2x2 grid):
#   4个特征 → 各占一个subplot
#   图例统一放在右侧（Case base hue + α-value shade）

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Patch

# ========= Paths =========
SAVE_DIR = Path("/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/out_dir")
CSV_PATH = SAVE_DIR / "analysis_ready_results_abs.csv"

# ========= Load data =========
df = pd.read_csv(CSV_PATH)

# ========= Columns / features =========
target = "SolveTime"
selected_features = ["sum_r_cu", "fillin", "tree_max_deg", "t"]
need_cols = set(["Case", "A_parameter", target] + selected_features)
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Case"] = df["Case"].astype(str)

# ========= 处理 alpha =========
allowed_alpha = np.array([0.0, 5.0, 4.0, 3.0, 2.0])
def canon_alpha(a):
    try:
        v = float(a)
    except Exception:
        return np.nan
    idx = np.abs(allowed_alpha - v).argmin()
    return float(allowed_alpha[idx])

df["alpha"] = df["A_parameter"].apply(canon_alpha)
df = df.dropna(subset=["alpha"]).copy()

# ========= Case → base hue mapping =========
CASE_BASE_COLORS = {
    "case2764": "tab:orange",  # 指定颜色
}
fallback_cycle = ["tab:blue", "tab:red", "tab:purple", "tab:brown",
                  "tab:gray", "tab:pink", "tab:olive", "tab:cyan"]

cases = df["Case"].unique().tolist()
case_to_base = {}
k = 0
for c in cases:
    if c in CASE_BASE_COLORS:
        case_to_base[c] = CASE_BASE_COLORS[c]
    else:
        case_to_base[c] = fallback_cycle[k % len(fallback_cycle)]
        k += 1

# ========= α-value → shade (mix with white) =========
SHADE_BY_ALPHA = {0.0: 0.0, 5.0: 0.1, 4.0: 0.3, 3.0: 0.5, 2.0: 0.7}
alpha_order = [0.0, 5.0, 4.0, 3.0, 2.0]

def mix_with_white(color_name, amount):
    r, g, b = to_rgb(color_name)
    return to_hex(((1-amount)*r + amount*1.0,
                   (1-amount)*g + amount*1.0,
                   (1-amount)*b + amount*1.0))

def color_for_alpha(case, alpha):
    base = case_to_base[case]
    shade = SHADE_BY_ALPHA.get(alpha, 1)
    return mix_with_white(base, shade)

# ========= Plot (2x2 grid) =========
plt.figure(figsize=(28, 12))

for i, feat in enumerate(selected_features, 1):
    ax = plt.subplot(2, 2, i)
    for c in cases:
        subc = df[df["Case"] == c]
        for a in alpha_order:
            sub = subc[subc["alpha"] == a]
            if sub.empty:
                continue
            ax.scatter(sub[feat], sub[target],
                       s=28, alpha=0.9, color=color_for_alpha(c, a))
    # Pearson correlation
    try:
        r = float(df[[feat, target]].corr().iloc[0, 1])
    except Exception:
        r = np.nan
    ax.set_xlabel(feat)
    ax.set_ylabel(target)
    ax.set_title(f"{feat} vs {target}\n(Pearson r={r:.2f})")

# ========= Global legend on the right =========
handles = []
for c in cases:
    for a in alpha_order:
        if ((df["Case"] == c) & (df["alpha"] == a)).any():
            facecolor = color_for_alpha(c, a)
            label = f"{c}, α={a}"
            handles.append(Patch(facecolor=facecolor, edgecolor="none", label=label))

plt.legend(handles=handles, title="Case + α value (shade)",
           bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=False)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # 给右边legend留空间

out_path = SAVE_DIR / "scatterplots_case_alpha_shades_2x2.png"
plt.savefig(out_path, dpi=200)
plt.show()
print(f"Saved: {out_path}")
