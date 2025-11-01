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

class MultiDimNormalizer:
    
    def __init__(self, mode: NormalizationMode = "zscore", eps: float = 1e-8):
        self.mode = mode
        self.eps = eps
        self.fitted = False
        self.normalizers = []  # 存储每个特征维度的标准化器
        self.num_features = None
        self.original_data_shape = None  # 记录单个样本的原始形状
        
    def fit(self, X: torch.Tensor):
        """
        X: [..., N_features] - 最后一维是特征维度
        """
        X = X.float()
        
        # 记录原始形状（除了batch维度）
        if X.dim() == 1:
            self.original_data_shape = torch.Size([])  # 标量
            self.num_features = 1
            X = X.unsqueeze(-1)  # [N] -> [N, 1]
        else:
            self.original_data_shape = X.shape[1:]  # 去掉batch维度
            self.num_features = X.shape[-1]  # 最后一维是特征数
        
        # 将所有非特征维度展开成样本：[..., N_features] -> [total_samples, N_features]
        X_reshaped = X.view(-1, self.num_features)  # [total_samples, N_features]
        
        print(f"🔍 MultiDimNormalizer fit: original shape {X.shape} -> reshaped {X_reshaped.shape}")
        print(f"🔍 num_features: {self.num_features}, original_data_shape: {self.original_data_shape}")
        
        self.normalizers = []
        
        # 为每个特征维度创建独立的标准化器
        for dim in range(self.num_features):
            dim_data = X_reshaped[:, dim:dim+1]  # [total_samples, 1]
            norm = GlobalNormalizer(self.mode, self.eps).fit(dim_data)
            self.normalizers.append(norm)
            
        self.fitted = True
        return self
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., N_features] - 可以是单个样本或批量样本
        """
        if not self.fitted:
            return x
            
        x = x.float()
        original_shape = x.shape
        
        # 确保输入的特征维度匹配
        if x.shape[-1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x.shape[-1]}")
        
        # 将输入reshape为 [total_samples, N_features]
        x_reshaped = x.view(-1, self.num_features)
        
        # 对每个特征维度分别变换
        transformed_features = []
        for dim in range(self.num_features):
            dim_data = x_reshaped[:, dim:dim+1]  # [total_samples, 1]
            transformed = self.normalizers[dim].transform(dim_data)
            transformed_features.append(transformed.squeeze(-1))  # [total_samples]
        
        # 重新组合特征
        result = torch.stack(transformed_features, -1)  # [total_samples, N_features]
        
        # 恢复原始形状
        return result.view(original_shape)

    @property 
    def mean(self):
        """返回所有特征维度的均值"""
        if not self.fitted or self.mode != "zscore":
            return None
        return torch.tensor([norm.mean.item() for norm in self.normalizers])
    
    @property
    def std(self):
        """返回所有特征维度的标准差"""
        if not self.fitted or self.mode != "zscore":
            return None
        return torch.tensor([norm.std.item() for norm in self.normalizers])

def normalize_inplace(samples: Sequence[dict[str, Any]], *,
                      mode: NormalizationMode = "zscore",
                      key: str = "global_vec",
                      strict: bool = True):
    has_any = any((key in s) for s in samples)
    if not has_any:
        return None
    
    data_list = []
    original_shapes = []
    for i, s in enumerate(samples):
        if key not in s:
            if strict:
                raise ValueError(f"Sample {i} missing '{key}' while others have it.")
            else:
                continue
        # 确保数据是 float32 并保持原始形状
        data = torch.as_tensor(s[key], dtype=torch.float32)
        original_shapes.append(data.shape)
        data_list.append(data)  # 不再flatten，保持原始形状

    if not data_list:
        return None
    
    # 堆叠所有样本：[N_samples, ...] 
    all_data = torch.stack(data_list, 0)
    print(f"🔍 Normalizing {key}: shape {all_data.shape}")
    
    # 使用多维标准化器 - 最后一维是特征维度
    norm = MultiDimNormalizer(mode).fit(all_data)

    # 对每个样本应用变换
    j = 0
    for s in samples:
        if key not in s: 
            continue

        sample_data = all_data[j]  
        transformed = norm.transform(sample_data.unsqueeze(0))  
        s[key] = transformed  
        j += 1
    
    return norm
