import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
CSV_PATH = "grad_log copy.csv"  

def load_log(path):
    df = pd.read_csv(path, header=0,
                     names=["epoch", "step", "param_name", "grad_norm", "update_norm", "param_norm", "lr"],
                     dtype={
                         "epoch": int,
                         "step": int,
                         "param_name": str,
                         "grad_norm": float,
                         "update_norm": float,
                         "param_norm": float,
                         "lr": float
                     })
    return df

def plot_global_norms(df, save_prefix=None):
    agg = df.groupby("epoch").agg(
        grad_mean=("grad_norm", "mean"),
        grad_max=("grad_norm", "max"),
        upd_mean=("update_norm", "mean"),
        upd_max=("update_norm", "max"),
        param_mean=("param_norm", "mean"),
        lr=("lr", "mean"),
    ).reset_index()

    # 1) grad / update / param 
    plt.figure(figsize=(8, 5))
    plt.plot(agg["epoch"], agg["grad_mean"], label="grad_mean")
    plt.plot(agg["epoch"], agg["upd_mean"], label="update_mean")
    plt.plot(agg["epoch"], agg["param_mean"], label="param_mean")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("norm (log scale)")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_global_norms.png", dpi=200)
    else:
        plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(agg["epoch"], agg["lr"])
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_lr.png", dpi=200)
    else:
        plt.show()


def plot_relative_update(df, save_prefix=None):
    """相对步长：update_norm / param_norm"""
    df = df.copy()
    df["rel_update"] = df["update_norm"] / (df["param_norm"] + 1e-12)

    agg = df.groupby("epoch")["rel_update"].agg(["mean", "max"]).reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(agg["epoch"], agg["mean"], label="rel_update_mean")
    plt.plot(agg["epoch"], agg["max"], label="rel_update_max")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("relative update (|Δθ| / |θ|)")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_rel_update.png", dpi=200)
    else:
        plt.show()


def plot_gcn_vs_head(df, save_prefix=None):
    """
    对比 GCN 参数 vs head 参数 的梯度大小
    默认规则：
      - 名字里包含 'gcns' 的算 GCN
      - 名字里包含 'head' 的算 head_array / head_logits
    可以按需要改 filter。
    """
    is_gcn = df["param_name"].str.contains("gcns")
    is_head = df["param_name"].str.contains("head")

    gcn_agg = df[is_gcn].groupby("epoch")["grad_norm"].mean().reset_index(name="gcn_grad_mean")
    head_agg = df[is_head].groupby("epoch")["grad_norm"].mean().reset_index(name="head_grad_mean")

    merged = gcn_agg.merge(head_agg, on="epoch", how="inner")

    plt.figure(figsize=(8, 5))
    plt.plot(merged["epoch"], merged["gcn_grad_mean"], label="GCN grad_mean")
    plt.plot(merged["epoch"], merged["head_grad_mean"], label="Head grad_mean")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("grad norm (log scale)")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_gcn_vs_head_grad.png", dpi=200)
    else:
        plt.show()


def plot_layer_heatmap(df, save_prefix=None):
    """
    可选：按 layer 画 grad_norm 的热力图
    假设 GCN 层名字形如 'gcns.0.lin.weight' / 'gcns.1.ln.bias'
    """
    def get_layer(name: str):
        if name.startswith("gcns"):
            parts = name.split(".")  # ['gcns','0','lin','weight']
            if len(parts) > 1 and parts[1].isdigit():
                return f"gcn_{parts[1]}"
        elif name.startswith("head"):
            return "head"
        else:
            return "other"

    df = df.copy()
    df["layer"] = df["param_name"].apply(get_layer)

    # 只看 gcn 层 & head
    mask = df["layer"].isin(["head"] + [f"gcn_{i}" for i in range(10)])  # 最多 10 层，按需改
    sub = df[mask]

    pivot = sub.pivot_table(
        index="epoch",
        columns="layer",
        values="grad_norm",
        aggfunc="mean",
    )

    values = np.log10(pivot.values + 1e-12)  # log10 防止数量级差太大

    plt.figure(figsize=(8, 5))
    im = plt.imshow(values, aspect="auto", origin="lower")
    plt.colorbar(im, label="log10(grad_norm)")
    plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns, rotation=45)
    plt.yticks([])  # step 太多就不画刻度了
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_layer_grad_heatmap.png", dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    df = load_log(CSV_PATH)

    # 1. 全局 norm + lr
    plot_global_norms(df, save_prefix="pic/appnp/gradlog")

    # 2. 相对更新量
    plot_relative_update(df, save_prefix="pic/appnp/gradlog")

    # 3. GCN vs head 梯度对比
    plot_gcn_vs_head(df, save_prefix="pic/appnp/gradlog")

    # 4. 按层 grad 热力图（可选）
    plot_layer_heatmap(df, save_prefix="pic/appnp/gradlog")