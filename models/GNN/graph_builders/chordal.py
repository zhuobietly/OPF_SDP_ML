import torch
from .base import GraphBuilder

class ChordalGraph(GraphBuilder):
    def build(self, raw):
        A = raw["A_chordal"]
        if A.dim() == 2:
            ei = A.nonzero(as_tuple=False).t().contiguous()
        else:
            ei = A
        meta = {"kind":"chordal", "alpha": raw.get("alpha"), "strategy": raw.get("strategy")}
        return {"edge_index": ei, "edge_attr": None, "global": meta}
