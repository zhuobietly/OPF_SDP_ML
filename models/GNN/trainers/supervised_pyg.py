import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from ..gcn_utils.logging import log

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)                 # (B, out_dim)
        loss = loss_fn(pred.view_as(data.y), data.y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        bs = data.y.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = loss_fn(pred.view_as(data.y), data.y)
        bs = data.y.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

def fit(model, train_ds, val_ds, epochs=50, batch_size=32, lr=3e-4, device="cpu", shuffle=True):
    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = PyGDataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best = float("inf")
    for ep in range(1, epochs+1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device)
        va = eval_epoch(model, val_loader,   loss_fn, device)
        log(f"[PyG] Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")
        if va < best: best = va
    return {"val_mse": best}
