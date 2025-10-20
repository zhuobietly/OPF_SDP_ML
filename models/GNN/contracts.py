# contracts.py
from typing import TypedDict, Optional
import torch

class Batch(TypedDict, total=False):
    # 可选/按需出现的键 —— 暂不强制你现在就提供
    A_hat: torch.Tensor        # (B, N, N)
    X: torch.Tensor            # (B, N, F)
    gvec: torch.Tensor         # (B, G)
    y_reg: torch.Tensor        # (B,)
    y_arr_reg: torch.Tensor    # (B, K) 或 padding 后的 (B, Kmax)
    y_cls: torch.Tensor        # (B,)
    mask: torch.Tensor         # (B, K) 变长时的有效掩码

class ModelOut(TypedDict, total=False):
    pred_y_reg: torch.Tensor   # (B,)
    pred_arr_reg: torch.Tensor # (B, K)
    logits: torch.Tensor       # (B, K) 需要分类时
