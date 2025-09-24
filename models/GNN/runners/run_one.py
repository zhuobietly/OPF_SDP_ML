
import os, sys, yaml, torch
from pathlib import Path
import random
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT /"configs" / "exp_toy.yaml"
#path :/home/goatoine/Documents/Lanyue/models/GNN/configs/exp_tol.yaml
from model_varients.registry import build as build_model
from graph_builders.original import OriginalGraph
try:
    from features.loads_phys_topo import LoadsPhysTopo
except ModuleNotFoundError:
    from features.loads_phys_topo import LoadsPhysTopo

from data_loader.dataset_opf import OPFGraphDataset, make_collate_fn
from trainers.supervised import fit
from gcn_utils.seed import set_seed
from gcn_utils.global_normalize import normalize_inplace
import pickle
import torch 
import numpy as np 
import logging

def setup_logging():
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/home/goatoine/Documents/Lanyue/models/GNN/run_log/model_outputs.log'),  # 保存到文件
        ]
    )
# === the kind of gcn model structure ===
BUILDERS = {
    "original": OriginalGraph,
    # "original_plus_stats": OriginalPlusStats,
    # "chordal": ChordalGraph,
}


def build_samples_debug_toy(num=80, minN=8, maxN=28, K=24):
    #这个如果再用要+一个变量y_arr_regss
    out = []
    for i in range(num):
        N = maxN
        # 邻接矩阵
        A = (torch.rand(N, N) > 0.82).float()
        A = torch.triu(A, 1); A = A + A.t()
        y_reg = torch.randn(K).abs() 
        y_cls = torch.argmin(y_reg).long()
        global_vec = torch.randn(16)
        raw = {
            "A": A,
            "node_load": torch.randn(N, 2),
            "degree": A.sum(dim=1, keepdim=True),
            "global_vec": global_vec,
            "y_reg": y_reg,
            "y_cls": y_cls,
        }
        out.append(raw)
    return out

def convert_samples_to_torch(samples):
    """将 numpy samples 转换为 torch samples"""
    torch_samples = []
    for sample in samples:
        torch_sample = {}
        for key, value in sample.items():
            if key == "y_cls":
                    torch_sample[key] = torch.tensor(value).long()
            elif key == "y_reg":
                    torch_sample[key] = torch.tensor(value).float()
            elif isinstance(value, np.ndarray): # y_cls 转为 long
                torch_sample[key] = torch.from_numpy(value).float()
            else:
                torch_sample[key] = value
        torch_samples.append(torch_sample)
    return torch_samples
def main():
    setup_logging()
    cfg = yaml.safe_load(open(CONFIG_PATH, "r"))
    set_seed(cfg.get("seed", 42))
    # samples = build_samples_debug_toy()
    with open("/home/goatoine/Documents/Lanyue/data/data_for_GCN/data_basic_GCN/samples.pkl", "rb") as f:
        samples = pickle.load(f)
    # 转换为 torch tensor
    samples = convert_samples_to_torch(samples)
    
    #samples = load_from_csv_or_jld2(...)
    # 3) Setup graph builder and feature pipeline
    builder = BUILDERS[cfg["builder"]]()
    pipeline = LoadsPhysTopo()
    mode_name = cfg.get("global",{}).get("norm","zscore")
    norm = normalize_inplace(samples, mode=mode_name, key="global_vec")

    # 4) split and generate Dataset (三份：train, val, test)
    # notes that the samples are already shuffled
    train_ratio = 0.6  # 60% 训练
    val_ratio = 0.2    # 20% 验证  
    test_ratio = 0.2   # 20% 测试

    train_split = int(train_ratio * len(samples))
    val_split = int((train_ratio + val_ratio) * len(samples))

    train_ds = OPFGraphDataset(samples[:train_split], builder.build, pipeline.node_features, norm)
    val_ds   = OPFGraphDataset(samples[train_split:val_split], builder.build, pipeline.node_features, norm)
    test_ds  = OPFGraphDataset(samples[val_split:], builder.build, pipeline.node_features, norm)

    # simple model, that every graph is has the same size, so we can batch them directly
    # collate_fn is used to merge a list of samples to a batch
    train_ds.collate_fn = make_collate_fn(train_ds)
    val_ds.collate_fn = make_collate_fn(val_ds)
    test_ds.collate_fn = make_collate_fn(test_ds)

    print(f"[INFO] 数据集分割:")
    print(f"  训练集: {len(train_ds)} 样本")
    print(f"  验证集: {len(val_ds)} 样本") 
    print(f"  测试集: {len(test_ds)} 样本")

    # 5) build model
    in_dim = train_ds[0]["X"].shape[1]
    model_args = {
    "name": cfg["model"]["name"],
    "in_dim": in_dim,
    "hidden": cfg["model"].get("hidden", [128, 128]),
    "out_dim": cfg["model"].get("out_dim", 24),
    "dropout": cfg["model"].get("dropout", 0.2),
    }
    if "g_dim" in cfg["model"]:
        model_args["g_dim"] = cfg["model"]["g_dim"]

    model = build_model(**model_args) 

    # ---- 调试建议：首次 batch 打印一下形状，确认输入无误 ----
    first = train_ds[0]
    print("[dbg] A_hat:", first["A_hat"].shape, "X:", first["X"].shape, "y_reg:", first["y_reg"].shape,"y_arr_reg:", first["y_arr_reg"].shape, "y_cls:", first["y_cls"].shape)

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
    # the metric of Y visualization
    print("Done.")


if __name__ == "__main__":
    main()
