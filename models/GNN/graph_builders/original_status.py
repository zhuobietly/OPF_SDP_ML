import torch
from .base import GraphBuilder

class OriginalPlusStats(GraphBuilder):
    def build(self, raw):
        A = raw["A"]
        if A.dim() == 2:
            ei = A.nonzero(as_tuple=False).t().contiguous()
        else:
            ei = A
        stats = raw.get("chordal_stats", {})
        return {"edge_index": ei, "edge_attr": None,
                "global": {"kind": "original+stats", "stats": stats}}
