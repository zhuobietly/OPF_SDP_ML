import torch
from .base import GraphBuilder

class OriginalGraph(GraphBuilder):
    def build(self, raw):
        A = raw["A"]
        if A.dim() == 2:
            # return [[index of start nodes], [index of end nodes]]
            ei = A.nonzero(as_tuple=False).t().contiguous()
        else:
            ei = A
        return {"edge_index": ei, "edge_attr": None, "global": {"kind": "original"}}
# can put the features of chordal graph to the global attributes
# can concat X = concat(X, repeat(g, N)),simple 
# or can add a virtual node and connect it to all other nodes