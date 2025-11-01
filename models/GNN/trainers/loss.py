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
    def __init__(self, w_arr=1.0, w_cls=3.0, tau=0.2):
        self.w_arr = w_arr
        self.w_cls = w_cls
        self.tau = tau

    def __call__(self, out, batch):
        loss = 0.0

        # --- 数组回归部分 ---
        if "pred_arr_reg" in out and "y_arr_reg" in batch:
            loss_arr = F.l1_loss(out["pred_arr_reg"], batch["y_arr_reg"])
            loss += self.w_arr * loss_arr
        else:
            loss_arr = 0.0

        # --- 分类部分（基于 pred_arr_reg 最小值索引）---
        if "pred_arr_reg" in out and "y_cls" in batch:
            logits = -out["pred_arr_reg"] / self.tau  # softmin logits
            loss_cls = F.cross_entropy(logits, batch["y_cls"])
            loss += self.w_cls * loss_cls
        else:
            loss_cls = 0.0

        return loss

def _cb_class_weights_from_counts(counts: torch.Tensor, beta: float,
                                  normalize: bool = True,
                                  zero_count_mode: str = "zero",
                                  eps: float = 1e-8):
    """
    Class-Balanced 权重（兼容 n_c=0）：
      - 仅对 n_c>0 的类计算 w_c = (1-β)/(1-β^{n_c})
      - n_c=0 的类按 zero_count_mode 处理：
          'zero' -> w_c = 0
          'min1' -> 计算时临时当作 n_c=1
          'eps'  -> 计算时临时当作 n_c=eps
      - 归一化只在 n_c>0 的子集中进行：sum(w_c * n_c) == sum(n_c)
    """
    device = counts.device
    counts = counts.to(dtype=torch.float32)
    pos_mask = counts > 0                              # 有样本的类
    any_pos = bool(pos_mask.any())

    counts_tilde = counts.clone()
    if zero_count_mode == "min1":
        counts_tilde = torch.where(pos_mask, counts_tilde, torch.ones_like(counts_tilde))
    elif zero_count_mode == "eps":
        counts_tilde = torch.where(pos_mask, counts_tilde, torch.full_like(counts_tilde, eps))
    else:  # "zero"
        pass

    if beta == 0.0:
        w = torch.ones_like(counts_tilde, device=device)
    else:
        w = torch.zeros_like(counts_tilde, device=device)
        if any_pos:
            eff = 1.0 - torch.pow(torch.tensor(beta, dtype=counts_tilde.dtype, device=device), counts_tilde[pos_mask])
            w_pos = (1.0 - beta) / eff.clamp_min(eps)  # 数值安全
            w[pos_mask] = w_pos
        if zero_count_mode == "zero":
            w[~pos_mask] = 0.0
        elif zero_count_mode in ("min1", "eps"):
            pass

    # 归一化（只在 n_c>0 的子集内），让平均权重 ≈ 1
    if normalize and any_pos:
        denom = (w[pos_mask] * counts[pos_mask]).sum().clamp_min(eps)
        scale = counts[pos_mask].sum() / denom
        w[pos_mask] = w[pos_mask] * scale
    return w

def _manual_ce_with_class_weights(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor ):
    """
      CE_i = -log softmax(logits_i)[y_i]
      loss = mean( w_{y_i} * CE_i )
    """
    # 数值稳定的 log-softmax
    z = logits - logits.max(dim=1, keepdim=True).values       # [B, C]
    logZ = torch.log(torch.exp(z).sum(dim=1, keepdim=True))   # [B, 1]
    log_probs = z - logZ                                      # [B, C]
    nll = -log_probs[torch.arange(target.shape[0], device=target.device), target]  # [B]

    if class_weights is not None:
        w = class_weights.to(logits.device)[target]           # [B]
        loss = (w * nll).mean()
    else:
        loss = nll.mean()
    return loss

@register("loss", "arr_cls_cbce")
class ArrClsCBCEManual:
    """
    数组回归 + Class-Balanced 加权的手写 CE（兼容 n_c=0）：
      - 数组回归：L1(pred_arr_reg, y_arr_reg) * w_arr
      - 分类：若 out 有 "logits" 用之；否则 logits = -pred_arr_reg / tau
              再做手写 CE，并按 CB 权重加权 * w_cls
    """
    def __init__(self, w_arr: float = 1.0, w_cls: float = 1.0,
                 tau: float = 1.0,
                 beta: float = 0.999,
                 class_counts = None,                 # 训练集每类计数（list / tensor），允许含 0
                 normalize_weights: bool = True,
                 zero_count_mode: str = "zero"):      # 'zero' | 'min1' | 'eps'
        self.w_arr = float(w_arr)
        self.w_cls = float(w_cls)
        self.tau = float(tau)
        self.beta = float(beta)
        self.zero_count_mode = zero_count_mode
        self.normalize_weights = normalize_weights

        if class_counts is None:
            self.class_weights = None
        else:
            counts = torch.as_tensor(class_counts, dtype=torch.long)
            self.class_weights = _cb_class_weights_from_counts(
                counts, beta=self.beta,
                normalize=self.normalize_weights,
                zero_count_mode=self.zero_count_mode)

    def __call__(self, out, batch):
        loss = 0.0

        # --- 数组回归（可选） ---
        if "pred_arr_reg" in out and "y_arr_reg" in batch and self.w_arr != 0.0:
            loss_arr = F.l1_loss(out["pred_arr_reg"], batch["y_arr_reg"])
            loss = loss + self.w_arr * loss_arr

        # --- 分类（手写 CE + 类权重） ---
        if "y_cls" in batch and self.w_cls != 0.0:
            logits = -out["pred_arr_reg"] / self.tau
            y = batch["y_cls"].to(dtype=torch.long, device=logits.device)
            cw = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss_cls = _manual_ce_with_class_weights(logits, y, cw)
            loss = loss + self.w_cls * loss_cls

        return loss

    
