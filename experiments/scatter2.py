# -*- coding: utf-8 -*-
# Scatter plots of SolveTime vs selected features (2x3 grid):
# - First 5 subplots: scatter per feature
# - 6th subplot: legends (Case base hue + α-value shade)
#
# Rule:
#   • Each Case uses a base hue (e.g., case2764 → orange, case1888 → blue; others auto-assigned).
#   • Within the same Case, α ∈ {0,2,3,5} is encoded by brightness:
#         α=0 (deep) , α=2 (medium-deep), α=3 (medium-light), α=5 (light)

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Patch

# ========= Paths (single place to change) =========
SAVE_DIR = Path("/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/out_dir")
CSV_PATH = SAVE_DIR / "analysis_ready_results_abs.csv"

# ========= Load data =========
df = pd.read_csv(CSV_PATH)

# ========= Columns / features =========
target = "SolveTime"
selected_features = ["sum_r_cu", "sep_max", "fillin", "t", "tree_max_deg"]
# Ensure required columns exist
need_cols = set(["Case", "A_parameter", target] + selected_features)
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ========= Case as string; alpha as canonical {0,2,3,5} =========
df["Case"] = df["Case"].astype(str)

# Map numeric alpha to {0,2,3,5} by nearest value (robust to floats like 3.0)
allowed_alpha = np.array([0.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0])
def canon_alpha(a):
    try:
        v = float(a)
    except Exception:
        return np.nan
    idx = np.abs(allowed_alpha - v).argmin()
    return int(allowed_alpha[idx])

df["alpha"] = df["A_parameter"]
df = df.dropna(subset=["alpha"]).copy()
df["alpha"] = df["alpha"]

# ========= Case → base hue mapping =========
CASE_BASE_COLORS = {
    "case2764": "tab:orange",  # 指定：橙色系
    "case1888rte": "tab:blue",    # 指定：蓝色系
}
fallback_cycle = ["tab:green", "tab:red", "tab:purple", "tab:brown",
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
SHADE_BY_ALPHA = {0: 0.2, 5: 0.3, 4.5: 0.4, 4.0: 0.5, 3.5: 0.6, 3.0: 0.7, 2.5: 0.8, 2.0: 0.9}
alpha_order = [0.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]

def mix_with_white(color_name, amount):
    r, g, b = to_rgb(color_name)
    return to_hex(((1-amount)*r + amount*1.0,
                   (1-amount)*g + amount*1.0,
                   (1-amount)*b + amount*1.0))

def color_for_alpha(case, alpha):
    base = case_to_base[case]
    shade = SHADE_BY_ALPHA.get(alpha, 0.8)
    return mix_with_white(base, shade)

# ========= Plot =========
plt.figure(figsize=(26, 10))

# 1..5: scatter subplots
for i, feat in enumerate(selected_features, 1):
    ax = plt.subplot(2, 3, i)
    for c in cases:
        subc = df[df["Case"] == c]
        # loop α in fixed order so shades are consistent
        for a in [0.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]:
            sub = subc[subc["alpha"] == a]
            if sub.empty:
                continue
            ax.scatter(sub[feat], sub[target],
                       s=28, alpha=0.9, color=color_for_alpha(c, a))
    # Pearson r (across all rows)
    try:
        r = float(df[[feat, target]].corr().iloc[0, 1])
    except Exception:
        r = np.nan
    ax.set_xlabel(feat)
    ax.set_ylabel(target)
    ax.set_title(f"{feat} vs {target}\n(Pearson r={r:.2f})")

# 6th subplot: legends
ax_leg = plt.subplot(2, 3, 6)
ax_leg.axis("off")

# Case legend (base hue)
case_handles = [Patch(facecolor=case_to_base[c], edgecolor="none", label=c) for c in cases]
# Arrange cases into rows if many
ncol_cases = 3 if len(case_handles) > 9 else min(len(case_handles), 3)
leg1 = ax_leg.legend(handles=case_handles, title="Case (base hue)",
                     loc="upper left", bbox_to_anchor=(0.0, 1.0),
                     ncol=ncol_cases, frameon=False)
ax_leg.add_artist(leg1)

# Alpha legend (shade levels) — use orange as representative base
rep_base = "tab:orange"
alpha_handles = [Patch(facecolor=mix_with_white(rep_base, SHADE_BY_ALPHA[a]),
                       edgecolor="none", label=f"α={a}") for a in alpha_order
                 if (df["alpha"] == a).any()]
ax_leg.legend(handles=alpha_handles, title="α value (shade)",
              loc="lower left", bbox_to_anchor=(0.0, 0.0),
              ncol=1, frameon=False)

# Case-Alpha legend (shade levels) — 每个case的每个α都展示
# 指定alpha顺序
alpha_order = [0.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]  # 0最深，2最浅
shade_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 0.2最深，0.8最浅（举例）
alpha_to_shade = dict(zip(alpha_order, shade_levels))

case_alpha_handles = []
for c in cases:
    for a in alpha_order:
        if ((df["Case"] == c) & (df["alpha"] == a)).any():
            facecolor = color_for_alpha(c, a)
            label = f"{c}, α={a}"
            case_alpha_handles.append(Patch(facecolor=facecolor, edgecolor="none", label=label))

ncol_legend = 2 if len(case_alpha_handles) > 10 else 1
ax_leg.legend(handles=case_alpha_handles, title="Case, α value (shade)",
              loc="lower left", bbox_to_anchor=(0.0, 0.0),
              ncol=ncol_legend, frameon=False)

plt.tight_layout()
out_path = SAVE_DIR / "scatterplots_case_alpha_shades.png"
plt.savefig(out_path, dpi=200)
plt.show()

print(f"Saved: {out_path}")
