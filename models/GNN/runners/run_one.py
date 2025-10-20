import os, sys, yaml, torch
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import random
CONFIG_PATH = PROJECT_ROOT /"configs" / "debug.yaml"

try:
    from features.loads_phys_topo import LoadsPhysTopo
except ModuleNotFoundError:
    from features.loads_phys_topo import LoadsPhysTopo

from gcn_utils.seed import set_seed
from gcn_utils.global_normalize import normalize_inplace
import pickle
import torch 
import numpy as np 
import logging
from registries import build
from torch.utils.data import DataLoader
from data_loader.datasets.dataset_opf import OPFGraphDataset, make_collate_fn

from data_loader import reader  # 这会触发 @register 装饰器
from model_varients import gcn_new
from trainers import task
from trainers import loss 

def setup_logging():
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/home/goatoine/Documents/Lanyue/models/GNN/run_log/model_outputs.log'),  # 保存到文件
        ]
    )

def save_checkpoint(model: torch.nn.Module, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)
    print(f"[CKPT] Saved to {path}")

def load_checkpoint(model_builder, model_args: dict, path: str, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = model_builder(**model_args)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[CKPT] Loaded from {path}")
    return model

def infer_in_dim_from_ds(ds):
    raw0 = ds.samples[0]                 # 直接拿原始样本，避免 batch/collate
    if ds.build_features is not None:    # 和 __getitem__ 同一逻辑，但不建图
        X0 = ds.build_features(raw0)
    else:
        X0 = raw0["X"]
    return X0.shape[-1]

def split_by_ratio(samples, train=0.6, val=0.2, test=0.2, seed=42, shuffle=True):
    assert abs(train + val + test - 1.0) < 1e-6
    idx = list(range(len(samples)))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
    n = len(samples)
    i_tr = int(train * n)
    i_va = int((train + val) * n)
    tr_idx, va_idx, te_idx = idx[:i_tr], idx[i_tr:i_va], idx[i_va:]
    # 切出子列表
    get = lambda ids: [samples[i] for i in ids]
    return get(tr_idx), get(va_idx), get(te_idx)

def main():
    setup_logging()
    cfg = yaml.safe_load(open(CONFIG_PATH, "r"))
    set_seed(cfg.get("seed", 42))
    reader   = build("reader", cfg["data"]["reader"])
    samples  = reader.load()                       # ← list[dict]，数量随 reader 变化

    # 2) 统一做比例切分（与你原来一致）
    train_ratio = 0.6
    val_ratio   = 0.2
    test_ratio  = 1.0 - train_ratio - val_ratio
    train_s, val_s, test_s = split_by_ratio(samples, train_ratio, val_ratio, test_ratio, seed=cfg["seed"], shuffle=True)
    # 3) 保持你原来的 Dataset/Collate 用法
    pipeline = LoadsPhysTopo()
    norm     = cfg["data"].get("norm", None)

    train_ds = OPFGraphDataset(train_s,  pipeline.node_features, norm)
    val_ds   = OPFGraphDataset(val_s,   pipeline.node_features, norm)
    test_ds  = OPFGraphDataset(test_s,  pipeline.node_features, norm)

    train_ds.collate_fn = make_collate_fn(train_ds)
    val_ds.collate_fn   = make_collate_fn(val_ds)   
    test_ds.collate_fn  = make_collate_fn(test_ds)

    print(f"[INFO] 数据集分割:")
    print(f"  训练集: {len(train_ds)} 样本")
    print(f"  验证集: {len(val_ds)} 样本") 
    print(f"  测试集: {len(test_ds)} 样本")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=True,
                              collate_fn=getattr(train_ds, "collate_fn", None))
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=False,
                              collate_fn=getattr(val_ds, "collate_fn", None))
    test_loader   = DataLoader(test_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=False,
                              collate_fn=getattr(test_ds, "collate_fn", None))
    # 试探 in_dim（X 的最后一维）


    # 用法：
    in_dim = infer_in_dim_from_ds(train_ds)
     
    # 1) 构建 model / loss / task
    model = build("model", "gcn_basic",
                in_dim=in_dim, hidden=64, layers=2,
                out_array_dim=cfg["model"].get("out_array_dim", 15))

    loss_fn = build("loss", "arr_cls")  # 或 "multihead_basic" 按需要换

    task = build("task", "opf_basic_task",
                model=model, loss_fn=loss_fn,
                device="cuda" if torch.cuda.is_available() else "cpu",
                lr=cfg.get("train", {}).get("lr", 1e-3))

    # 2) 训练 + 验证（可选）
    epochs = cfg.get("train", {}).get("epochs", 5)
    for ep in range(epochs):
        tr_loss = task.train_one_epoch(train_loader)
        val_loss = task.validate(val_loader)
        print(f"Epoch {ep} | loss={tr_loss:.4f} | {val_loss:.4f}")

                # 你的 evaluate 里要的配置原样给它

    # 3) 测试：走你现有的 evaluate.py
    test_metrics = task.test_with_evaluator(test_loader, cfg)
    print("[TEST]", test_metrics)

if __name__ == "__main__":
    main()
