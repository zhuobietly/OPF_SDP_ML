import torch
from torch.utils.data import DataLoader
from .gcn_utils.logging import log

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        A, X, y = batch["A_hat"].to(device), batch["X"].to(device), batch["y"].to(device)
        pred = model(A, X)
        loss = loss_fn(pred.view_as(y), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for batch in loader:
        A, X, y = batch["A_hat"].to(device), batch["X"].to(device), batch["y"].to(device)
        pred = model(A, X)
        loss = loss_fn(pred.view_as(y), y)
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)

def fit(model, train_ds, val_ds, epochs=50, batch_size=8, lr=3e-4, device="cpu"):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=getattr(train_ds, "collate_fn", None))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=getattr(val_ds, "collate_fn", None))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best = float("inf")
    for ep in range(1, epochs+1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device)
        va = eval_epoch(model, val_loader, loss_fn, device)
        log(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")
        if va < best: best = va
    return {"val_mse": best}
