import torch, torch.nn.functional as F
from registries import register

@register("loss", "scalar_mse")
class ScalarMSE:
    def __init__(self):
        pass
    def __call__(self, out, batch):
        return F.mse_loss(out["pred_y_reg"], batch["y_reg"], reduction="mean")

@register("loss", "multihead_basic")
class MultiheadBasic:
    def __init__(self, w_arr=1.0, w_sca=0.5, w_cls=1.0):
        self.w_arr = w_arr
        self.w_sca = w_sca
        self.w_cls = w_cls
    
    def __call__(self, out, batch):
        loss = 0.0
        if "pred_arr_reg" in out and "y_arr_reg" in batch:
            loss += self.w_arr * F.l1_loss(out["pred_arr_reg"], batch["y_arr_reg"])
        if "pred_y_reg" in out and "y_reg" in batch:
            loss += self.w_sca * F.mse_loss(out["pred_y_reg"], batch["y_reg"])
        if "logits" in out and "y_cls" in batch:
            loss += self.w_cls * F.cross_entropy(out["logits"], batch["y_cls"])
        return loss

@register("loss", "arr_cls")
class ArrCls:
    def __init__(self, w_arr=1.0, w_cls=1.0):
        self.w_arr = w_arr
        self.w_cls = w_cls
    
    def __call__(self, out, batch):
        loss = 0.0
        if "pred_arr_reg" in out and "y_arr_reg" in batch:
            loss += self.w_arr * F.l1_loss(out["pred_arr_reg"], batch["y_arr_reg"])
        if "logits" in out and "y_cls" in batch:
            loss += self.w_cls * F.cross_entropy(out["logits"], batch["y_cls"])
        return loss