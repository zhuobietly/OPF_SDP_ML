import torch
import torch.nn.functional as F

def multitask_loss(logits, y_vec, y_cls, lam: float = 0.5):
    """
        logits: (B,K)  Model output (no need to apply softmax first)
        y_vec:  (B,K)  Multi-output regression targets
        y_cls:  (B,)   Classification targets (LongTensor, values 0..K-1)
        lambda:        Regression weight (0~1)

        Returns: total loss, regression MSE, classification CE
    """
    # regret：MSE
    loss_reg = F.mse_loss(logits, y_vec)

    # classification: CrossEntropyLoss internally applies log-softmax to logits
    loss_cls = F.cross_entropy(logits, y_cls)

    loss = lam * loss_reg + (1.0 - lam) * loss_cls
    #loss_reg.detach(), loss_cls.detach()
    return loss
