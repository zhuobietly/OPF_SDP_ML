import torch.nn as nn

class GNNBase(nn.Module):
    def __init__(self, in_dim:int, hidden:list[int], out_dim:int,
                 dropout:float=0.0, activ:str="relu"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.dropout = dropout
        self.activ = activ

    def forward(self, *args, **kwargs):
        raise NotImplementedError
