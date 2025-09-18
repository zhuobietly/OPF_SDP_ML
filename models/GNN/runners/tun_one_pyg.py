"""
不等尺寸图（PyG）最简 runner：
- 直接 python models/GNN/runners/run_simple_pyg.py
- 不用手写 PYTHONPATH：脚本里已注入项目根到 sys.path
- 你可以在 build_samples_debug_toy / dataset_pyg / gcn_pyg forward 处打断点
"""
import os, sys, yaml, torch
from pathlib import Path

# 让“直接运行脚本”也能 import models.GNN.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "models" / "GNN" / "configs" / "exp_pyg.yaml"

from models.GNN.model_variants.gcn import build as build_model   # 注册表入口
from models.GNN.model_variants import gcn_pyg                    # 确保 GCN_PYG 被 import 注册
from models.GNN.graph_builders.original import OriginalGraph
try:
    from models.GNN.features.loads_phys_topo import LoadsPhysTopo
except ModuleNotFoundError:
    from models.GNN.feaatures.loads_phys_topo import LoadsPhysTopo

from models.GNN.data_loader.dataset_pyg import OPFGraphDatasetPyG
from models.GNN.trainers.supervised_pyg import fit
from models.GNN.gcn_utils.seed import set_seed

def build_samples_debug_toy(num=60, minN=6, maxN=18):
    """随机生成不等尺寸图（调通流程用）。替换成你从 Julia 导出的 raw 列表即可。"""
    import random
    out = []
    for i in range(num):
        N = random.randint(minN, maxN)
        A = (torch.rand(N, N) > 0.82).float()
        A = torch.triu(A, 1); A = A + A.t()
        raw = {
            "A": A,                                             # (N,N)
            "node_load": torch.randn(N, 2),                     # for features
            "degree": A.sum(dim=1, keepdim=True),
            "label": torch.tensor([float(i % 5)], dtype=torch.float32),  # graph-level label
        }
        out.append(raw)
    return out

def main():
    cfg = yaml.safe_load(open(CONFIG_PATH, "r"))
    set_seed(cfg.get("seed", 42))

    # 数据（不等尺寸 toy；换成你自己的 raw 加载）
    samples = build_samples_debug_toy(num=80, minN=8, maxN=32)

    # 构图 + 特征
    builder = OriginalGraph()
    pipeline = LoadsPhysTopo()

    # Dataset（返回 Data）
    def build_graph(raw): return builder.build(raw)
    def build_features(g, raw): return pipeline.node_features(g, raw)

    split = int(0.8 * len(samples))
    train_ds = OPFGraphDatasetPyG(samples[:split], build_graph, build_features)
    val_ds   = OPFGraphDatasetPyG(samples[split:], build_graph, build_features)

    # 模型（从第一条样本估 in_dim）
    in_dim = train_ds[0].x.shape[1]
    model = build_model(
        cfg["model"]["name"],     # "GCN_PYG"
        in_dim=in_dim,
        hidden=cfg["model"].get("hidden", [128,128]),
        out_dim=cfg["model"].get("out_dim", 1),
        dropout=cfg["model"].get("dropout", 0.2),
    )

    # 调试：看一眼尺寸
    d0 = train_ds[0]
    print(f"[dbg] N={d0.num_nodes}, x={tuple(d0.x.shape)}, E={d0.num_edges}, y={tuple(d0.y.shape)}")

    # 训练
    fit(
        model,
        train_ds, val_ds,
        epochs=cfg["train"].get("epochs", 5),
        batch_size=cfg["train"].get("batch_size", 8),
        lr=cfg["train"].get("lr", 3e-4),
        device="cpu",
    )
    print("Done.")

if __name__ == "__main__":
    main()
