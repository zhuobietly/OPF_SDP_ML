
import os, sys, yaml, torch
from pathlib import Path

# === 让“直接运行脚本”也能找到 models.GNN 包 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # 指向包含 models/ 的目录
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# === 这里写默认配置路径，直接改这一行就能换实验 ===
CONFIG_PATH = PROJECT_ROOT /"configs" / "exp_toy.yaml"
#path :/home/goatoine/Documents/Lanyue/models/GNN/configs/exp_tol.yaml
# === 统一导入（注意：如果你的文件夹叫 feaatures，请先改名为 features）===
from model_varients.registry import build as build_model
from graph_builders.original import OriginalGraph
try:
    from features.loads_phys_topo import LoadsPhysTopo
except ModuleNotFoundError:
    from features.loads_phys_topo import LoadsPhysTopo

from data_loader.dataset_opf import OPFGraphDataset, collate_fn
from trainers.supervised import fit
from gcn_utils.seed import set_seed

# === the kind of gcn model structure ===
BUILDERS = {
    "original": OriginalGraph,
    # "original_plus_stats": OriginalPlusStats,
    # "chordal": ChordalGraph,
}


def build_samples_debug_toy(num=60, N=10):
    """最小可跑的 toy 数据（你先用它验证流程；之后换成真实 raw 数据加载）"""
    def make_sample(label):
        A = (torch.rand(N, N) > 0.8).float()
        A = torch.triu(A, 1); A = A + A.t()
        raw = {
            "A": A,                                    # (N,N) 0/1 邻接
            "node_load": torch.randn(N, 2),            # 节点特征示例
            "degree": A.sum(dim=1, keepdim=True),      # 度
            "label": torch.tensor([label], dtype=torch.float32),  # 回归标签
        }
        return raw
    return [make_sample(float(i % 5)) for i in range(num)]


def main():
    cfg = yaml.safe_load(open(CONFIG_PATH, "r"))
    set_seed(cfg.get("seed", 42))
    samples = build_samples_debug_toy(num=60, N=10)
    #samples = load_from_csv_or_jld2(...)
    # 3) 选择图构建 & 特征流水线（这里可以打断点，看 raw/g 的结构）
    builder = BUILDERS[cfg["builder"]]()
    pipeline = LoadsPhysTopo()

    # 4) 切分数据并创建 Dataset
    split = int(0.8 * len(samples))
    train_ds = OPFGraphDataset(samples[:split], builder.build, pipeline.node_features)
    val_ds   = OPFGraphDataset(samples[split:], builder.build, pipeline.node_features)
    # 简单 collate（等大小图）。变尺寸图后续可换 PyG 的 Batch
    train_ds.collate_fn = collate_fn
    val_ds.collate_fn = collate_fn

    # 5) 构建模型
    in_dim = train_ds[0]["X"].shape[1]
    model = build_model(
        cfg["model"]["name"],
        in_dim=in_dim,
        hidden=cfg["model"].get("hidden", [128, 128]),
        out_dim=1,
        dropout=cfg["model"].get("dropout", 0.2),
    )

    # ---- 调试建议：首次 batch 打印一下形状，确认输入无误 ----
    first = train_ds[0]
    print("[dbg] A_hat:", first["A_hat"].shape, "X:", first["X"].shape, "y:", first["y"].shape)

    # 6) 训练
    fit(
        model,
        train_ds,
        val_ds,
        epochs=cfg["train"].get("epochs", 5),
        batch_size=cfg["train"].get("batch_size", 8),
        lr=cfg["train"].get("lr", 3e-4),
        device="cpu",
    )

    print("Done.")


if __name__ == "__main__":
    main()
