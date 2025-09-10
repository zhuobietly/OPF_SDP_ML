# -*- coding: utf-8 -*-
# Scatter plots of SolveTime vs selected features (2x3 grid):
# - First 5 subplots: scatter per feature
# - 6th subplot: legends (Case color families + Method shades)
#
# Color rule:
#   Each Case has a base hue (e.g., case2764=orange, case1888=blue, others auto-assigned).
#   Within the same Case, AMD is darkest, MD medium, MFI lightest.

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

# ========= Config =========
selected_features = ["sum_r_cu", "sep_max", "fillin", "ActiveLimits", "tree_max_deg"]
target = "SolveTime"

# ========= Parse chordal method from columns =========
def parse_method(s: str) -> str:
    s = str(s)
    if re.search(r"\bAMD\b|Chordal_AMD", s, re.I): return "AMD"
    if re.search(r"\bMFI\b|Chordal_MFI", s, re.I): return "MFI"
    if re.search(r"\bMD\b|Chordal_MD",  s, re.I): return "MD"
    return "Other"

if "Strategy" in df.columns:
    df["Method"] = df["Strategy"].apply(parse_method)
else:
    df["Method"] = df["Formulation"].apply(parse_method)

df["Case"] = df["Case"].astype(str)

# ========= Case → base hue mapping =========
CASE_BASE_COLORS = {
    "case2764": "tab:orange",  # 橙色系（例：2764）
    "case1888": "tab:blue",    # 蓝色系（例：1888）
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

# ========= Method → shade (mix with white) =========
SHADE_BY_METHOD = {"AMD": 0.10, "MD": 0.35, "MFI": 0.65, "Other": 0.50}  # 0=deep, 1=white

def mix_with_white(color_name, amount):
    r, g, b = to_rgb(color_name)
    return to_hex(((1-amount)*r + amount*1.0,
                   (1-amount)*g + amount*1.0,
                   (1-amount)*b + amount*1.0))

def color_for(case_name, method_name):
    base = case_to_base.get(str(case_name), "black")
    amt  = SHADE_BY_METHOD.get(method_name, 0.50)
    return mix_with_white(base, amt)

# ========= Plot =========
plt.figure(figsize=(26, 10))
axes = []

# 1..5: scatter subplots
for i, feat in enumerate(selected_features, 1):
    ax = plt.subplot(2, 3, i)
    axes.append(ax)
    for c in cases:
        subc = df[df["Case"] == c]
        for m in ["AMD", "MD", "MFI", "Other"]:
            sub = subc[subc["Method"] == m]
            if sub.empty:
                continue
            ax.scatter(sub[feat], sub[target],
                       s=28, alpha=0.9, color=color_for(c, m))
    # Pearson r (over all rows)
    try:
        r = float(df[[feat, target]].corr().iloc[0, 1])
    except Exception:
        r = np.nan
    ax.set_xlabel(feat)
    ax.set_ylabel(target)
    ax.set_title(f"{feat} vs {target}\n(Pearson r={r:.2f})")

# 6th subplot (legend box)
ax_leg = plt.subplot(2, 3, 6)
ax_leg.axis("off")

# Case legend (base hues)
case_handles = [Patch(facecolor=case_to_base[c], edgecolor="none", label=c) for c in cases]
ncol_cases = 3 if len(case_handles) > 9 else min(len(case_handles), 3)
leg1 = ax_leg.legend(handles=case_handles, title="Case (base color)",
                     loc="upper left", bbox_to_anchor=(0.0, 1.0),
                     ncol=ncol_cases, frameon=False)
ax_leg.add_artist(leg1)

# Case-Method legend (shade)
method_labels = [m for m in ["AMD", "MD", "MFI"] if m in df["Method"].unique()]
case_method_handles = []
for c in cases:
    for m in method_labels:
        facecolor = color_for(c, m)
        label = f"{c}_{m}"
        case_method_handles.append(Patch(facecolor=facecolor, edgecolor="none", label=label))

ax_leg.legend(handles=case_method_handles, title="Case_Method (shade)",
              loc="lower left", bbox_to_anchor=(0.0, 0.0),
              ncol=2, frameon=False)

plt.tight_layout()
out_path = SAVE_DIR / "scatterplots.png"
plt.savefig(out_path, dpi=200)
plt.show()

print(f"Saved: {out_path}")
