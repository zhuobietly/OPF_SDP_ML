from __future__ import annotations
from typing import Literal, Sequence, Any
import torch

NormalizationMode = Literal["zscore", "minmax", "robust", "none"]

class GlobalNormalizer:
    def __init__(self, mode: NormalizationMode = "zscore", eps: float = 1e-8):
        assert mode in ("zscore","minmax","robust","none")
        self.mode, self.eps, self.fitted = mode, eps, False
        self.mean = self.std = self.minv = self.maxv = self.median = self.iqr = None

    def fit(self, X: torch.Tensor):
        X = X.float()
        if self.mode == "none":
            self.fitted = True; return self
        if self.mode == "zscore":
            self.mean = X.mean(0)
            self.std  = X.std(0, unbiased=False).clamp_min(self.eps)
        elif self.mode == "minmax":
            self.minv = X.min(0).values
            self.maxv = X.max(0).values
            same = (self.maxv - self.minv).abs() < self.eps
            self.maxv[same] = self.minv[same] + 1.0
        elif self.mode == "robust":
            self.median = X.median(0).values
            q75, q25 = X.quantile(0.75, dim=0), X.quantile(0.25, dim=0)
            self.iqr = (q75 - q25).clamp_min(self.eps)
        self.fitted = True
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if (self.mode == "none") or (not self.fitted):
            return x
        if self.mode == "zscore": return (x - self.mean) / self.std
        if self.mode == "minmax": return (x - self.minv) / (self.maxv - self.minv)
        if self.mode == "robust": return (x - self.median) / self.iqr
        return x

def normalize_inplace(samples: Sequence[dict[str, Any]], *,
                      mode: NormalizationMode = "zscore",
                      key: str = "global_vec",
                      strict: bool = True):
    has_any = any((key in s) for s in samples)
    if not has_any:
        return None
    g_list = []
    for i, s in enumerate(samples):
        if key not in s:
            if strict:
                raise ValueError(f"Sample {i} missing '{key}' while others have it.")
            else:
                continue
        g_list.append(torch.as_tensor(s[key], dtype=torch.float32).view(-1))
    if not g_list:
        return None
    all_g = torch.stack(g_list, 0)
    norm = GlobalNormalizer(mode).fit(all_g)
    # j = 0
    # # for s in samples:
    # #     if key not in s: continue
    # #     s[key] = norm.transform(all_g[j]).tolist()
    # #     j += 1
    return norm
