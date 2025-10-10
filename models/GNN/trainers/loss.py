# import torch
# import torch.nn.functional as F

# def multitask_loss(pred_arr_reg, pred_y_reg, y_arr_reg, y_reg, y_cls, lam: float = 0.5):
#     """
#         logits: (B,K)  Model output (no need to apply softmax first)
#         y_vec:  (B,K)  Multi-output regression targets
#         y_cls:  (B,)   Classification targets (LongTensor, values 0..K-1)
#         lambda:        Regression weight (0~1)
#         Returns: total loss, regression MSE, classification CE
#     """
#     # regret：MSE
#     loss_arr_reg = F.mse_loss(pred_arr_reg, y_arr_reg)
#     loss_reg = F.mse_loss(pred_y_reg, y_reg)

#     # classification: CrossEntropyLoss internally applies log-softmax to logits
#     #loss_cls = F.cross_entropy(logits, y_cls)
#     loss_cls = F.cross_entropy(-pred_arr_reg, y_cls)

#     loss = lam * loss_reg + (1.0 - lam) * loss_cls
#     return loss

import torch
import torch.nn.functional as F

# def multitask_loss(pred_arr_reg, pred_y_reg, y_arr_reg, y_reg, y_cls,
#                    w_arr=1.0, w_sca=0.5, w_cls=2.0, tau=0.5):
#     # 1) 数组回归（锚定 pred_arr_reg，防止塌缩）
#     loss_arr = F.mse_loss(pred_arr_reg, y_arr_reg)

#     # 2) 标量回归（需要就用；不需要可把 w_sca 设为 0）
#     loss_sca = F.mse_loss(pred_y_reg, y_reg)

#     # 3) 分类（只看相对大小，去掉行偏置）
#     logits = - (pred_arr_reg - pred_arr_reg.mean(dim=1, keepdim=True)) / tau
#     loss_cls = F.cross_entropy(logits, y_cls)

#     return loss_sc
def multitask_loss(pred_y_reg, y_reg,
                   w_arr=1.0, w_sca=0.5, w_cls=2.0, tau=0.5):
    # 2) 标量回归（需要就用；不需要可把 w_sca 设为 0）
    loss_sca = F.mse_loss(pred_y_reg, y_reg)

    return loss_sca