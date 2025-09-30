import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from gcn_utils.logging import log
from trainers.loss import multitask_loss

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        if model.__class__.__name__ == "GCNGlobal":
            A, X, y_reg, y_arr_reg, y_cls, gvec = batch["A_hat"].to(device), batch["X"].to(device), batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device), batch["gvec"].to(device)
            pred_arr_reg, pred_y_reg, pred_y_cls = model(A, X, gvec)
        else:
            A, X, y_reg, y_arr_reg, y_cls = batch["A_hat"].to(device), batch["X"].to(device), batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device)
            pred_arr_reg, pred_y_reg, pred_y_cls = model(A, X)
        loss = loss_fn(pred_arr_reg, pred_y_reg, y_arr_reg, y_reg, y_cls)
        optimizer.zero_grad(); 
        loss.backward(); 
        # === 这里加梯度体检 ===
        lr = next(iter(optimizer.param_groups))["lr"]
        for name, p in model.named_parameters():
            if p.grad is not None and ("scalar_head" in name or "lastmlp" in name):
                g = p.grad.detach()
                w = p.detach()
                g_abs_mean = g.abs().mean().item()
                g_abs_max  = g.abs().max().item()
                w_abs_mean = w.abs().mean().item()
                rel_update = lr * g_abs_mean / (w_abs_mean + 1e-12)
                print(f"[grad] {name:30s} "
                      f"|w|mean={w_abs_mean:.3e}  "
                      f"|g|mean={g_abs_mean:.3e}  |g|max={g_abs_max:.3e}  "
                      f"rel_update≈{rel_update:.3e}")
        # =====================
        optimizer.step()
        total += loss.item() * y_reg.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for batch in loader:
        if model.__class__.__name__ == "GCNGlobal":
            A, X, y_reg, y_arr_reg, y_cls, gvec = batch["A_hat"].to(device), batch["X"].to(device), batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device), batch["gvec"].to(device)
            pred_arr_reg, pred_y_reg, pred_y_cls = model(A, X, gvec)
        else:
            A, X, y_reg, y_arr_reg, y_cls = batch["A_hat"].to(device), batch["X"].to(device), batch["y_reg"].to(device), batch["y_arr_reg"].to(device), batch["y_cls"].to(device)
            pred_arr_reg, pred_y_reg, pred_y_cls = model(A, X)
        loss = loss_fn(pred_arr_reg, pred_y_reg, y_arr_reg, y_reg, y_cls)
        total += loss.item() * y_reg.size(0)
    return total / len(loader.dataset)

def fit(model, train_ds, val_ds, epochs=50, batch_size=8, lr=3e-4, device="cpu", plot_path="./models/GNN/runners/loss_curve.png"):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=getattr(train_ds, "collate_fn", None))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=getattr(val_ds, "collate_fn", None))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = multitask_loss
    best = float("inf")

    # 存 loss
    train_losses, val_losses = [], []

    for ep in range(1, epochs+1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device)
        va = eval_epoch(model, val_loader, loss_fn, device)
        train_losses.append(tr)
        val_losses.append(va)

        log(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")
        if va < best: best = va

    # ---- 绘图 ----
    plt.figure()
    plt.plot(range(epochs-5, epochs+1), train_losses[-6:], label="train_loss")
    plt.plot(range(epochs-5, epochs+1), val_losses[-6:], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    print(f"✅ Loss curve saved to {plot_path}")

    return {"val_mse": best}
