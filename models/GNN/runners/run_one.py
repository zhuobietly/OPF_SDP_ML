import yaml, torch
from model.gcn import build as build_model
from graph_builders.original import OriginalGraph
from features.loads_phys_topo import LoadsPhysTopo
from dataio.dataset_opf import OPFGraphDataset, collate_fn
from trainers.supervised import fit
from utils.seed import set_seed

BUILDERS = {"original": OriginalGraph}

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))

    # --- toy samples (替换为 Julia 产出的 raw 数据) ---
    N = 10
    def make_sample(label):
        A = (torch.rand(N, N) > 0.8).float()
        A = torch.triu(A,1); A = A + A.t()
        raw = {"A": A,
               "node_load": torch.randn(N,2),
               "degree": A.sum(dim=1, keepdim=True),
               "label": torch.tensor([label], dtype=torch.float32)}
        return raw

    samples = [make_sample(float(i%5)) for i in range(60)]
    split = int(0.8*len(samples))
    builder = BUILDERS[cfg["builder"]]()
    pipeline = LoadsPhysTopo()

    train_ds = OPFGraphDataset(samples[:split], builder.build, pipeline.node_features)
    val_ds   = OPFGraphDataset(samples[split:], builder.build, pipeline.node_features)
    train_ds.collate_fn = collate_fn
    val_ds.collate_fn = collate_fn

    in_dim = train_ds[0]["X"].shape[1]
    model = build_model(cfg["model"]["name"],
                        in_dim=in_dim,
                        hidden=cfg["model"].get("hidden",[128,128]),
                        out_dim=1,
                        dropout=cfg["model"].get("dropout",0.2))

    fit(model, train_ds, val_ds,
        epochs=cfg["train"].get("epochs", 5),
        batch_size=cfg["train"].get("batch_size", 8),
        lr=cfg["train"].get("lr", 3e-4),
        device="cpu")

if __name__ == "__main__":
    import sys; main(sys.argv[1])
