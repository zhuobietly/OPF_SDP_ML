import torch
from .base import FeaturePipeline

class LoadsPhysTopo(FeaturePipeline):
    def node_features(self, graph_dict, raw):
        Xs = []
        # to add more features, just append to Xs
        if "node_load" in raw: Xs.append(raw["node_load"].float())
        if "degree" in raw:    Xs.append(raw["degree"].float())
        # if we broadcast some graph-level stats to nodes
        if "broadcast_stats" in raw: Xs.append(raw["broadcast_stats"].float())
        if len(Xs) == 0:
            raise ValueError("No node features found")
        return torch.cat(Xs, dim=1)
