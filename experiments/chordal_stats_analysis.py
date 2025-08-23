# -*- coding: utf-8 -*-
"""
绝对时间（无baseline）版：把 Strategy 定义为 (Formulation, Merge, A_parameter)
生成图：
  - 每个 Case：A 箱线(按 Strategy)；B ΣP/ΣQ vs log(time)；C 结构指标 vs log(time)
  - ALL 合并：ALL_A / ALL_B / ALL_C
输入：
  --in_dir  : 放所有 CSV 与 JSON 的目录
  --out_dir : 输出表与图片的目录
"""
import argparse, os, re, glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")             # 无图形界面也能保存图片
import matplotlib.pyplot as plt

# ---------------- 工具函数 ----------------
def parse_perturbation(x):
    """从 CSV 的 Perturbation 字段解析 (k, seed)。兼容 '(0.07, 70)'/ '0.07,70' 等"""
    if pd.isna(x): return (np.nan, np.nan)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
    if len(nums) >= 2:
        return (float(nums[0]), int(float(nums[1])))
    return (np.nan, np.nan)

def _normalize_case(case_str: str) -> str:
    """
    归一化 Case 名称：
    - 去掉 pglib_opf_ / pglib- / opf_ 等前缀
    - 返回以 'case' 开头的主体，如 case118、case1951rte、case2746wop
    """
    s = (case_str or "").strip()
    s = re.sub(r'^(pglib[_-])?opf[_-]', '', s, flags=re.IGNORECASE)
    m = re.search(r'(case[0-9]+[a-zA-Z]*.*)$', s, flags=re.IGNORECASE)
    return m.group(1) if m else s

def find_json(in_dir: Path, case, k, seed, id_):
    """
    更鲁棒的 JSON 匹配（不区分大小写）：
    - case 统一到 'caseXXX...' 形式
    - k 同时尝试 g / .2f / .3f
    - seed 与 id 强制转 int 字符串（70.0 -> '70'）
    - 多层 glob 兜底
    """
    norm = _normalize_case(str(case))
    # k 多格式
    try:
        kf = float(k)
        k_strs = [format(kf, 'g'), f"{kf:.2f}", f"{kf:.3f}"]
    except Exception:
        k_strs = [str(k)]
    # seed/id 转整数
    def _to_int_str(x):
        try:
            xi = int(round(float(x)))
            return str(xi)
        except Exception:
            return str(x)
    seed_str = _to_int_str(seed)
    id_str   = _to_int_str(id_)
    # 严格候选
    for ks in k_strs:
        p = in_dir / f"{norm}_{ks}_perturbation_{seed_str}_{id_str}.json"
        if p.exists(): return p
    # 宽松：k 任意但 seed/id 固定
    hits = list(in_dir.glob(f"{norm}_*perturbation_{seed_str}_{id_str}.json"))
    if hits: return hits[0]
    # 更宽松：包含 norm 与 'perturbation_seed_id'
    hits = [p for p in in_dir.glob("*.json")
            if norm in p.name and f"perturbation_{seed_str}_{id_str}" in p.name]
    if hits: return hits[0]
    print(f"[找不到JSON] case='{case}'→'{norm}', k={k}, seed={seed}→{seed_str}, id={id_}→{id_str}")
    return None

def load_load_stats(path: Path):
    """从 JSON 读出负荷特征"""
    with open(path, "r") as f:
        data = json.load(f)
    loads = data.get("load", {})
    Pd = [float(rec.get("pd", 0.0)) for rec in loads.values()]
    Qd = [float(rec.get("qd", 0.0)) for rec in loads.values()]
    P_sum = float(np.sum(Pd)) if Pd else np.nan
    Q_sum = float(np.sum(Qd)) if Qd else np.nan
    P_cv  = float(np.std(Pd)/np.mean(Pd)) if (Pd and np.mean(Pd)!=0) else np.nan
    P_max_frac = float(np.max(Pd)/np.sum(Pd)) if (Pd and np.sum(Pd)!=0) else np.nan
    return dict(P_sum=P_sum, Q_sum=Q_sum, P_cv=P_cv, P_max_frac=P_max_frac, json=path.name)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_log_time(arr):
    """对 SolveTime 取 log，非正的丢弃为 NaN"""
    a = np.asarray(arr, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = a > 0
    out[mask] = np.log(a[mask])
    return out

# ---- Strategy 合成 ----
def make_strategy(row):
    f = str(row.get("Formulation", ""))
    mflag = str(row.get("Merge", "")).strip().lower()
    m = "M=True" if mflag in ("true","1","t","yes","y") else "M=False"
    if m == "M=True":
        a = row.get("A_parameter", None)
        label_a = "—" if (a is None or (isinstance(a, float) and math.isnan(a))) else f"{float(a):g}"
        return f"{f} | {m} | α={label_a}"
    else:
        return f"{f} | {m}"

# ---------- 画图工具 ----------
def binned_line(ax, x, y, label, bins=6):
    """分位数分箱，连中位数折线"""
    ok = ~(np.isnan(x) | np.isnan(y))
    x = np.asarray(x[ok]); y = np.asarray(y[ok])
    if len(x) < max(10, bins):
        return False
    edges = np.quantile(x, np.linspace(0,1,bins+1))
    edges = np.maximum.accumulate(edges)  # 去重
    idx = np.digitize(x, edges[1:-1], right=True)
    xs, ys = [], []
    for b in range(bins):
        m = (idx==b)
        if m.sum() >= 3:
            xs.append(np.median(x[m]))
            ys.append(np.median(y[m]))
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=label)
        return True
    return False

def box_A_case(df, case, out_path):
    """图 A：单 Case，Strategy vs 绝对 log(SolveTime)（箱线+叠点）"""
    sub = df[(df["Case"] == case) & df["log_time"].notna()].copy()
    if sub.empty:
        print(f"[跳过] {case} 没有可用 log_time")
        return
    strategies = sorted(sub["Strategy"].unique().tolist())
    data = [sub.loc[sub["Strategy"]==s, "log_time"].values for s in strategies]
    plt.figure(figsize=(max(7.5, 1.2*len(strategies)), 4.8))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(strategies)+1), strategies, rotation=18)
    plt.ylabel("log(SolveTime)")
    plt.title(f"[A] Absolute time by Strategy — {case}")
    xs = [strategies.index(s)+1 for s in sub["Strategy"]]
    plt.scatter(xs, sub["log_time"], s=12, alpha=0.35)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    print(f"[图] {out_path}")

def pq_B_case(df, case, out_path):
    """图 B：单 Case，ΣP/ΣQ vs 绝对 log(SolveTime)，按 Strategy 画分箱中位线"""
    sub = df[(df["Case"] == case) & df["log_time"].notna()
             & df["P_sum"].notna() & df["Q_sum"].notna()].copy()
    if sub.empty:
        print(f"[跳过] {case} 没有可用 ΣP/ΣQ 或 log_time")
        return
    strategies = sorted(sub["Strategy"].unique().tolist())
    fig, axes = plt.subplots(1,2, figsize=(12,4.8), sharey=True)
    has1 = has2 = False
    for s in strategies:
        g = sub[sub["Strategy"]==s]
        has1 |= binned_line(axes[0], g["P_sum"], g["log_time"], s)
        has2 |= binned_line(axes[1], g["Q_sum"], g["log_time"], s)
    axes[0].set_xlabel("ΣP_load"); axes[0].set_ylabel("log(SolveTime)")
    axes[1].set_xlabel("ΣQ_load")
    axes[0].set_title(f"[B1] {case} — ΣP vs log(time)")
    axes[1].set_title(f"[B2] {case} — ΣQ vs log(time)")
    if has1 or has2:
        axes[1].legend(title="Strategy", loc="best")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    print(f"[图] {out_path}")

def struct_C_case(df, case, struct_col, out_path):
    """图 C：单 Case，结构指标 vs 绝对 log(SolveTime)（散点，按 Strategy 上色）"""
    if struct_col not in df.columns:
        print(f"[跳过] {case} 缺少结构列 {struct_col}")
        return
    sub = df[(df["Case"] == case) & df[struct_col].notna() & df["log_time"].notna()].copy()
    if sub.empty:
        print(f"[跳过] {case} 没有 {struct_col} 或 log_time")
        return
    strategies = sorted(sub["Strategy"].unique().tolist())
    plt.figure(figsize=(8.0,5.0))
    has_label = False
    colors = {s: plt.cm.tab20(i % 20) for i, s in enumerate(strategies)}
    for s in strategies:
        g = sub[sub["Strategy"]==s]
        if len(g) == 0: continue
        plt.scatter(g[struct_col], g["log_time"], s=16, alpha=0.65, label=s, color=colors[s])
        has_label = True
    plt.xlabel(struct_col); plt.ylabel("log(SolveTime)")
    plt.title(f"[C] {case} — {struct_col} vs log(SolveTime)")
    if has_label:
        plt.legend(markerscale=1.1, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    print(f"[图] {out_path}")

# ---------- 主流程 ----------
def main(in_dir: Path, out_dir: Path):
    ensure_dir(out_dir)
    fig_dir = out_dir / "figs"; ensure_dir(fig_dir)

    # 1) 合并 CSV
    csv_paths = sorted(glob.glob(str(in_dir / "*.csv")))
    if not csv_paths:
        raise SystemExit(f"[错误] 输入目录没有 CSV：{in_dir}")
    df_list = []
    for p in csv_paths:
        try:
            tmp = pd.read_csv(p)
            tmp["__source_csv__"] = os.path.basename(p)
            df_list.append(tmp)
        except Exception as e:
            print(f"[跳过] 读 {p} 失败：{e}")
    df = pd.concat(df_list, ignore_index=True)
    print(f"[信息] 合并 CSV 行数：{len(df)}")

    # 2) 解析扰动、场景键
    if "Perturbation" not in df.columns: df["Perturbation"] = None
    df[["pert_k","pert_seed"]] = df["Perturbation"].apply(parse_perturbation).apply(pd.Series)
    if "ID" not in df.columns: df["ID"] = -1
    for need in ["Case","Formulation","SolveTime","Merge","A_parameter"]:
        if need not in df.columns:
            raise SystemExit(f"[错误] CSV 缺少列：{need}")
    df["scenario_key"] = (
        df["Case"].astype(str) + "|" +
        df["pert_k"].astype(str) + "|" +
        df["pert_seed"].astype(str) + "|" +
        df["ID"].astype(str)
    )

    # 3) 合成 Strategy
    df["Strategy"] = df.apply(make_strategy, axis=1)

    # 4) 读取负荷 JSON（ΣP/ΣQ）
    P_sum, Q_sum, P_cv, P_max_frac, J = [], [], [], [], []
    for _, r in df.iterrows():
        jp = find_json(in_dir, str(r["Case"]), r["pert_k"], r["pert_seed"], r["ID"])
        if jp and jp.exists():
            st = load_load_stats(jp)
            P_sum.append(st["P_sum"]); Q_sum.append(st["Q_sum"])
            P_cv.append(st["P_cv"]);   P_max_frac.append(st["P_max_frac"]); J.append(st["json"])
        else:
            P_sum.append(np.nan); Q_sum.append(np.nan)
            P_cv.append(np.nan);  P_max_frac.append(np.nan); J.append(None)
    df["P_sum"] = P_sum; df["Q_sum"] = Q_sum
    df["P_cv"] = P_cv;   df["P_max_frac"] = P_max_frac; df["__load_json__"] = J

    # 5) 绝对 log 时间
    df["log_time"] = safe_log_time(df["SolveTime"])

    # 6) 结构列可用性
    struct_col = "sum_r_cu" if "sum_r_cu" in df.columns else ("fillin" if "fillin" in df.columns else None)

    # 7) 导出汇总表
    out_csv = out_dir / "analysis_ready_results_abs.csv"
    df.to_csv(out_csv, index=False)
    print(f"[信息] 已导出表：{out_csv}")

    # 8) 诊断：JSON 匹配与有效时间
    n_total = len(df)
    n_match = df["P_sum"].notna().sum()
    n_pos_time = np.isfinite(df["log_time"]).sum()
    print(f"[匹配统计] 负荷 JSON 命中：{n_match}/{n_total}，有效 SolveTime(>0)：{n_pos_time}/{n_total}")
    miss_json = df[df["P_sum"].isna()][["Case","Strategy","Perturbation","ID","__source_csv__"]]
    if len(miss_json):
        print("[诊断] 未匹配到 JSON（前 12 行）：")
        print(miss_json.head(12).to_string(index=False))

    # 9) 单个 Case 的 ABC 图
    cases = sorted(df["Case"].dropna().unique().tolist())
    for case in cases:
        box_A_case(df, case, fig_dir / f"{case}_A_box_abs_time.png")
        pq_B_case(df, case, fig_dir / f"{case}_B_PQ_vs_abs_time.png")
        if struct_col is not None:
            struct_C_case(df, case, struct_col, fig_dir / f"{case}_C_struct_vs_abs_time.png")
        else:
            print("[提示] 未检测到 sum_r_cu/fillin，图C跳过。")

    # 10) ALL 合并 ABC 图
    df_all = df.dropna(subset=["log_time"]).copy()
    if df_all.empty:
        print("[跳过] 没有有效 log_time，ALL 图不生成。")
        return

    # ALL_A：箱线 + 叠点（marker 按 Case）
    strategies = sorted(df_all["Strategy"].unique().tolist())
    plt.figure(figsize=(max(9.5, 1.2*len(strategies)), 5.4))
    data = [df_all.loc[df_all["Strategy"]==s, "log_time"].values for s in strategies]
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(strategies)+1), strategies, rotation=18)
    plt.ylabel("log(SolveTime)")
    plt.title("[ALL] Absolute time by Strategy — all cases")
    markers = ['o','s','^','D','P','X','*','v']
    case_list = sorted(df_all["Case"].dropna().unique().tolist())
    case2m = {c: markers[i % len(markers)] for i, c in enumerate(case_list)}
    for c in case_list:
        sub = df_all[df_all["Case"]==c]
        xs = [strategies.index(s)+1 for s in sub["Strategy"]]
        plt.scatter(xs, sub["log_time"], s=12, alpha=0.3, marker=case2m[c], label=str(c))
    if len(case_list) > 0:
        plt.legend(title="Case", markerscale=1.3, ncol=min(4, len(case_list)))
    plt.tight_layout(); plt.savefig(fig_dir / "ALL_A_box_abs_time.png", dpi=160); plt.close()
    print(f"[图] {fig_dir / 'ALL_A_box_abs_time.png'}")

    # ALL_B：ΣP/ΣQ vs log_time（按 Strategy 画分箱中位线）
    need_cols = ["P_sum","Q_sum","log_time","Strategy"]
    if all(col in df_all.columns for col in need_cols):
        fig, axes = plt.subplots(1,2, figsize=(12.5,5.0), sharey=True)
        has1 = has2 = False
        for s in strategies:
            g = df_all[df_all["Strategy"]==s]
            has1 |= binned_line(axes[0], g["P_sum"], g["log_time"], s)
            has2 |= binned_line(axes[1], g["Q_sum"], g["log_time"], s)
        axes[0].set_xlabel("ΣP_load"); axes[0].set_ylabel("log(SolveTime)")
        axes[1].set_xlabel("ΣQ_load")
        axes[0].set_title("[ALL-B1] ΣP vs log(time)")
        axes[1].set_title("[ALL-B2] ΣQ vs log(time)")
        if has1 or has2:
            axes[1].legend(title="Strategy", loc="best")
        plt.tight_layout(); plt.savefig(fig_dir / "ALL_B_PQ_vs_abs_time.png", dpi=160); plt.close()
        print(f"[图] {fig_dir / 'ALL_B_PQ_vs_abs_time.png'}")
    else:
        print("[ALL] 缺少 P_sum/Q_sum，ALL_B 跳过。")

    # ALL_C：结构 vs log_time（按 Strategy 上色、Case 为不同 marker）
    struct_col = "sum_r_cu" if "sum_r_cu" in df_all.columns else ("fillin" if "fillin" in df_all.columns else None)
    if struct_col is not None:
        sub = df_all.dropna(subset=[struct_col])
        if not sub.empty:
            plt.figure(figsize=(9.0,5.4))
            color_map = {s: plt.cm.tab20(i % 20) for i, s in enumerate(strategies)}
            case2m = {c: markers[i % len(markers)] for i, c in enumerate(case_list)}
            for (s, c), g in sub.groupby(["Strategy","Case"]):
                plt.scatter(g[struct_col], g["log_time"], s=16,
                            color=color_map.get(s, None),
                            marker=case2m.get(c,'o'), alpha=0.7,
                            label=f"{s} | {c}")
            plt.xlabel(struct_col); plt.ylabel("log(SolveTime)")
            plt.title(f"[ALL] {struct_col} vs log(SolveTime)")
            # legend 去重（组合多时可以注释掉）
            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            if len(uniq) <= 24:
                plt.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)
            plt.tight_layout(); plt.savefig(fig_dir / "ALL_C_struct_vs_abs_time.png", dpi=160); plt.close()
            print(f"[图] {fig_dir / 'ALL_C_struct_vs_abs_time.png'}")
        else:
            print("[ALL] 没有结构列或全是 NaN，ALL_C 跳过。")
    else:
        print("[ALL] 未检测到 sum_r_cu/fillin，ALL_C 跳过。")

    print(f"[完成] 图片输出目录：{fig_dir}")
    print(f"[完成] 汇总表：{out_csv}")

# ---------------- 入口 ----------------
if __name__ == "__main__":
    IN_DIR  = Path("/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/in_dir")
    OUT_DIR = Path("/home/goatoine/Documents/Lanyue/data/chordal_analysis_dataset/out_dir")
    main(IN_DIR, OUT_DIR)
