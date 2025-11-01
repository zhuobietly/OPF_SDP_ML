import os, sys, yaml, torch
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import random
CONFIG_PATH = PROJECT_ROOT /"configs" / "basic_newloss.yaml"

try:
    from features.loads_phys_topo import LoadsPhysTopo
except ModuleNotFoundError:
    from features.loads_phys_topo import LoadsPhysTopo

from gcn_utils.seed import set_seed
import pickle
import torch 
import numpy as np 
from registries import build
from torch.utils.data import DataLoader
from data_loader.datasets.dataset_opf import OPFGraphDataset, make_collate_fn

from data_loader import reader  
from model_varients import gcn_new
from trainers import task
from trainers import loss 
from gcn_utils.normalize import GlobalNormalizer, normalize_inplace

import torch

def infer_class_counts_from_list(samples, num_classes: int):
    """
    从样本列表统计每类样本数（允许为0）。
    每个元素可为 dict（含键 y_cls/label/y）或直接是张量/数值。
    支持索引标签 [B] 或 one-hot/logits [B, C]（自动 argmax）。
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    key_candidates = ("y_cls", "label", "y")

    for sample in samples:
        # 提取标签
        if isinstance(sample, dict):
            y = None
            for k in key_candidates:
                if k in sample:
                    y = sample[k]
                    break
        else:
            y = sample  # 直接是标签值

        y = torch.as_tensor(y)

        if y.ndim == 0:
            # 标量：直接视为单一样本类别索引
            idx = y.unsqueeze(0).to(torch.long)
        elif y.ndim == 1:
            # [B]：类别索引
            idx = y.to(torch.long)
        elif y.ndim == 2:
            # [B, C]：one-hot/logits
            idx = y.argmax(dim=1).to(torch.long)
        else:
            raise ValueError(f"Unsupported label shape: {tuple(y.shape)}")

        # 截断范围
        idx = idx.clamp(min=0, max=num_classes - 1)

        # 累加计数
        bc = torch.bincount(idx, minlength=num_classes)
        counts[:len(bc)] += bc.to(torch.long)

    return counts.tolist()


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
    raw0 = ds.samples[0]                 
    if ds.build_features is not None:    
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
    get = lambda ids: [samples[i] for i in ids]
    return get(tr_idx), get(va_idx), get(te_idx)

def main():
    cfg = yaml.safe_load(open(CONFIG_PATH, "r"))
    set_seed(cfg.get("seed", 42))
    reader   = build("reader", cfg["data"]["reader"])
    samples  = reader.load() 
    norm_style     = cfg["data"].get("norm", None)     
    #replace the data with normalized ones                
    norm = normalize_inplace(samples, mode=norm_style, key="node_load") if norm_style else None

    train_ratio = 0.6
    val_ratio   = 0.2
    test_ratio  = 1.0 - train_ratio - val_ratio
    train_s, val_s, test_s = split_by_ratio(samples, train_ratio, val_ratio, test_ratio, seed=cfg["seed"], shuffle=True)

    pipeline = LoadsPhysTopo()
    

    train_ds = OPFGraphDataset(train_s,  pipeline.node_features)
    val_ds   = OPFGraphDataset(val_s,   pipeline.node_features)
    test_ds  = OPFGraphDataset(test_s,  pipeline.node_features)
    

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

    in_dim = infer_in_dim_from_ds(train_ds)
     
    # model / loss / task
    model = build("model", "gcn_basic",
                in_dim=in_dim, hidden=64, layers=2,
                out_array_dim=cfg["model"].get("out_array_dim", 15))
    
    num_classes = cfg["model"].get("out_array_dim", 15)
    class_counts_auto = infer_class_counts_from_list(train_s, num_classes)

    print(f"[INFO] 训练集每类样本数 (C={num_classes}): {class_counts_auto}")

    #loss_fn = build("loss", "arr_cls")  
    loss_fn = build("loss", cfg["loss"]["name"],
                    w_arr=cfg["loss"].get("w_arr", 1.0),
                    w_cls=cfg["loss"].get("w_cls", 1.0),
                    tau=cfg["loss"].get("tau", 1),
                    beta=cfg["loss"].get("beta", 0.99),
                    class_counts=class_counts_auto,
                    normalize_weights=cfg["loss"].get("normalize_weights", True))

    task = build("task", "opf_basic_task",
                model=model, loss_fn=loss_fn,
                device="cpu",
                lr=cfg.get("train", {}).get("lr", 1e-3), log_dir =cfg.get("logger", {}).get("log_dir"))

    epochs = cfg.get("train", {}).get("epochs", 5)
    for ep in range(epochs):
        tr_loss = task.train_one_epoch(train_loader)
        val_loss = task.validate(val_loader)
        print(f"Epoch {ep} | loss={tr_loss:.4f} | {val_loss:.4f}")
        
    task.plot_loss_curves(cfg)
    task.save_loss_data(cfg)
    test_metrics = task.test_with_evaluator(test_loader, cfg)
    print("[TEST]", test_metrics)

if __name__ == "__main__":
    main()

