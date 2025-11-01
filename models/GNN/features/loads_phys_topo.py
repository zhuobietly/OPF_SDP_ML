import torch
from .base import FeaturePipeline

class LoadsPhysTopo(FeaturePipeline):
    def node_features(self, raw):
        Xs = []
        # to add more features, just append to Xs
        if "node_load" in raw: 
            # [1, 2746, 2] -> [2746, 2]
            node_load = raw["node_load"].float()
            if node_load.dim() == 3 and node_load.shape[0] == 1:
                node_load = node_load.squeeze(0)
            Xs.append(node_load)
        if "degree" in raw:    Xs.append(raw["degree"].float())
        # if we broadcast some graph-level stats to nodes
        if "broadcast_stats" in raw: Xs.append(raw["broadcast_stats"].float())
        if len(Xs) == 0:
            raise ValueError("No node features found")
        return torch.cat(Xs, dim=1)
