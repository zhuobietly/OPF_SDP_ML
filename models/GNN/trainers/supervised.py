# ===== minimal_train.py (drop-in replacement) =====
import os, logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gcn_utils.logging import log
from trainers.loss import multitask_loss

# ========== 统一且精简的 logger ==========
def setup_training_logger(log_path: str, level=logging.INFO) -> logging.Logger:
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_minimal")
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"=== Log start {datetime.now().isoformat(sep=' ', timespec='seconds')} ===")
    return logger

def group_grad_report(model, logger, tag="", groups=("gnn.", "scalar_head.", "att_q.", "mlp_24.")):
    """
    汇总每个前缀组的梯度范数总和 & 参数数量
    在 loss.backward() 之后、optimizer.step() 之前调用（最后一个 batch 调一次即可）
    """
    for gpref in groups:
        tot, cnt, none = 0.0, 0, 0
        for name, p in model.named_parameters():
            if not (name.startswith(gpref) and p.requires_grad): 
                continue
            if p.grad is None:
                none += 1
                continue
            tot += float(p.grad.detach().norm())
            cnt += 1
        logger.info(f"[{tag}] GRAD[{gpref}] total={tot:.3e} params={cnt} gradNone={none}")



def show_modules(logger, model, prefix=None):
    for name, m in model.named_modules():
        if name == "":  # 跳过根
            continue
        if prefix and not (name == prefix or name.startswith(prefix + ".")):
            continue
        depth = name.count(".")
        logger.info("  " * depth + f"{name} -> {m.__class__.__name__}")

def _grad_stats(model):
    grads = []
    none_cnt = zero_cnt = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            none_cnt += 1
            continue
        gnorm = torch.linalg.vector_norm(p.grad.detach()).item()
        if gnorm == 0.0:
            zero_cnt += 1
        grads.append((gnorm, n))
    grads.sort(reverse=True, key=lambda x: x[0])
    total = sum(v for v, _ in grads)
    top2 = grads[:2]
    return total, none_cnt, zero_cnt, top2

# ========== 训练/评估 ==========
def train_epoch(model, loader, optimizer, loss_fn, device, ep, logger, debug=False):
    model.train()
    total_loss_sum = 0.0
    total_items = 0

    last_grad_info = None
    num_batches = len(loader)  # 为了判断最后一个 batch

    for i, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)

        if model.__class__.__name__ == "GCNGlobal":
            A, X = batch["A_hat"].to(device), batch["X"].to(device)
            y_reg, y_arr_reg, y_cls = batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device)
            gvec = batch["gvec"].to(device)
            pred_arr_reg, pred_y_reg, pred_y_cls = model(A, X, gvec)
        else:
            A, X = batch["A_hat"].to(device), batch["X"].to(device)
            # y_reg, y_arr_reg, y_cls = batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device)
            y_reg = batch["y_reg"].to(device)
            # pred_arr_reg, pred_y_reg, pred_y_cls = model(A, X)
            pred_y_reg = model(A, X)

        loss = loss_fn(pred_y_reg, y_reg)

        loss.backward()

        # —— 只在最后一个 batch 记录一次分组梯度（放在 backward 之后、step 之前）
        # if i == num_batches:
        #     group_grad_report(model, logger, tag=f"epoch{ep}")

        if debug:
            last_grad_info = _grad_stats(model)

        optimizer.step()

        bs = y_reg.size(0)
        total_loss_sum += loss.item() * bs
        total_items += bs

    avg_loss = total_loss_sum / max(1, total_items)

    lr0 = optimizer.param_groups[0].get("lr", None) if optimizer.param_groups else None
    msg = f"[epoch{ep}] lr={lr0} train_loss={avg_loss:.6f}"
    if debug and last_grad_info is not None:
        total, none_cnt, zero_cnt, top2 = last_grad_info
        top_str = ", ".join([f"{n}: {v:.2e}" for v, n in top2])
        msg += f" | grad_total={total:.3e} gradNone={none_cnt} gradZero={zero_cnt} | top2 {{{top_str}}}"
    logger.info(msg)
    return avg_loss


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, ep, logger):
    model.eval()
    total_loss_sum = 0.0
    total_items = 0
    for batch in loader:
        if model.__class__.__name__ == "GCNGlobal":
            A, X = batch["A_hat"].to(device), batch["X"].to(device)
            y_reg, y_arr_reg, y_cls = batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device)
            gvec = batch["gvec"].to(device)
            pred_arr_reg, pred_y_reg, pred_y_cls = model(A, X)
        else:
            A, X = batch["A_hat"].to(device), batch["X"].to(device)
            y_reg, y_arr_reg, y_cls = batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device)
            pred_y_reg = model(A, X)

        loss = loss_fn(pred_y_reg, y_reg)
        bs = y_reg.size(0)
        total_loss_sum += loss.item() * bs
        total_items += bs

    avg_loss = total_loss_sum / max(1, total_items)
    logger.info(f"[epoch{ep}] val_loss={avg_loss:.6f}")
    return avg_loss

def fit(model, train_ds, val_ds, epochs=50, batch_size=8, lr=3e-4, device="cpu",
        plot_path="./models/GNN/runners/loss_curve.png",
        log_path="logs/train.log",
        DEBUG_GRAD=False,   # ← 仅当需要时才记录梯度概况
        PRINT_MODEL_ONCE=False,  # ← 仅当需要时打印模型/子模块
        PRINT_PREFIX=None   # 例如 "att_q"
        ):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=getattr(train_ds, "collate_fn", None))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=getattr(val_ds, "collate_fn", None))

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = multitask_loss
    best = float("inf")

    logger = setup_training_logger(log_path)

    if PRINT_MODEL_ONCE:
        logger.info("=== Model outline ===")
        show_modules(logger, model, prefix=PRINT_PREFIX)

    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device, ep, logger, debug=DEBUG_GRAD)
        va = eval_epoch(model, val_loader, loss_fn, device, ep, logger)

        train_losses.append(tr); val_losses.append(va)
        log(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")  
        if va < best: best = va

    # ---- 绘图（最后 6 个点，长度不足时自动适配）----
    k = min(6, len(train_losses))
    xs = list(range(len(train_losses) - k + 1, len(train_losses) + 1))
    plt.figure()
    plt.plot(xs, train_losses[-k:], label="train_loss")
    plt.plot(xs, val_losses[-k:], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    Path(os.path.dirname(plot_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=180)
    print(f"✅ Loss curve saved to {plot_path}")

    return {"val_mse": best}
