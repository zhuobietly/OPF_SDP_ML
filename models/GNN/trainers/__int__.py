# 统一对外暴露训练相关入口
from .supervised import train_epoch, fit  # 如果 supervised.py 里有 fit
from .loss import multitask_loss

__all__ = ["train_epoch", "fit", "multitask_loss"]
