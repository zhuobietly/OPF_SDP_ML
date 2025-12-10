import torch, torch.nn as nn
from registries import register

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)
    def forward(self, A_hat, X):
        # A_hat: [Batch, N, N], X: [Batch, N, F]
        # 如果 A_hat 是密集矩阵，转换为稀疏格式
        if A_hat.ndim == 3:  # Batched dense matrix
            B, N, F = X.shape
            H_list = []
            for i in range(B):
                A_sparse = A_hat[i].to_sparse()
                H_i = torch.sparse.mm(A_sparse, X[i])  # [N, F]
                H_list.append(H_i)
            H = torch.stack(H_list, dim=0)  # [B, N, F]
        else:
            H = torch.matmul(A_hat, X)
        
        H = self.lin(H)
        return self.drop(self.act(self.ln(H)))
    
class GraphConvOneTwoHop(nn.Module):
    """
    α1 A_hat + α2 A_hat^2 to approximate g_hat
    forward(A_hat, X)
        A_hat: [B, N, N] 
        X    : [B, N, F_in]
    """
    def __init__(self, in_dim, out_dim, dropout=0.0,
                 alpha1_init=1.0, alpha2_init=1.0, learnable_alpha=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)

        if learnable_alpha:
            self.alpha1 = nn.Parameter(torch.tensor(alpha1_init, dtype=torch.float32))
            self.alpha2 = nn.Parameter(torch.tensor(alpha2_init, dtype=torch.float32))
            self.alpha3 = nn.Parameter(torch.tensor(alpha2_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha1", torch.tensor(alpha1_init, dtype=torch.float32))
            self.register_buffer("alpha2", torch.tensor(alpha2_init, dtype=torch.float32))
            self.register_buffer("alpha3", torch.tensor(alpha2_init, dtype=torch.float32))
    def forward(self, A_hat, X):
        # A_hat: [B, N, N] 
        # X    : [B, N, F] 
        if A_hat.ndim == 3:  # batched dense -> per-batch sparse
            B, N, F = X.shape
            H1_list, H2_list, H3_list = [], [], []
            for i in range(B):
                A_sparse = A_hat[i].to_sparse()
                H1_i = torch.sparse.mm(A_sparse, X[i])   # 1-hop: A X
                H2_i = torch.sparse.mm(A_sparse, H1_i)
                H3_i = torch.sparse.mm(A_sparse, H2_i)
                H1_list.append(H1_i)
                H2_list.append(H2_i)
                H3_list.append(H3_i)
            H1 = torch.stack(H1_list, dim=0)             # [B, N, F]
            H2 = torch.stack(H2_list, dim=0)             # [B, N, F]
            H3 = torch.stack(H3_list, dim=0)             # [B, N, F]

        H_mix = self.alpha1 * H1 + self.alpha2 * H2 + self.alpha3 * H3     # (α1 A + α2 A^2 + α3 A^3) X
        H = self.lin(H_mix)
        H = self.ln(H)
        H = self.act(H)
        H = self.drop(H)
        return H

class GraphConvNHop(nn.Module):
    """
    Use α1 A_hat + α2 A_hat^2 + ... + αn A_hat^n to approximate g_hat
    forward(A_hat, X)
        A_hat: [B, N, N] 
        X    : [B, N, F_in]
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, n_hops=3,
                 alpha_init=1.0, learnable_alpha=True):
        """
        Args:
            in_dim: input feature dimension
            out_dim: output feature dimension
            dropout: dropout rate
            n_hops: number of hops (1, 2, 3, ..., n)
            alpha_init: initial value for all alphas
            learnable_alpha: whether to learn alpha parameters
        """
        super().__init__()
        self.n_hops = n_hops
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)

        # ✅ 创建 n 个 alpha 参数
        if learnable_alpha:
            self.alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
                for _ in range(n_hops)
            ])
        else:
            self.alphas = nn.ModuleList()
            for i in range(n_hops):
                self.register_buffer(f"alpha{i+1}", torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, A_hat, X):
        """
        A_hat: [B, N, N] 
        X    : [B, N, F] 
        """
        if A_hat.ndim == 3:  # batched dense -> per-batch sparse
            B, N, F = X.shape
            
            # ✅ 初始化 H_lists：每一跳一个列表
            H_lists = [[] for _ in range(self.n_hops)]
            
            for i in range(B):
                A_sparse = A_hat[i].to_sparse()
                
                # ✅ 循环计算 1-hop, 2-hop, ..., n-hop
                H_prev = X[i]  # 从输入特征开始
                for hop in range(self.n_hops):
                    H_curr = torch.sparse.mm(A_sparse, H_prev)  # A^hop X
                    H_lists[hop].append(H_curr)
                    H_prev = H_curr  # 用于下一跳计算
            
            # ✅ 将每一跳的结果堆叠成 [B, N, F]
            H_hops = [torch.stack(H_list, dim=0) for H_list in H_lists]
        
        else:
            # 非批处理版本（如果需要）
            H_hops = []
            H_prev = X
            for hop in range(self.n_hops):
                H_curr = torch.matmul(A_hat, H_prev)
                H_hops.append(H_curr)
                H_prev = H_curr

        # ✅ 加权混合所有跳数：α1*H1 + α2*H2 + ... + αn*Hn
        if isinstance(self.alphas, nn.ParameterList):
            # learnable_alpha=True
            H_mix = sum(alpha * H for alpha, H in zip(self.alphas, H_hops))
        else:
            # learnable_alpha=False
            H_mix = sum(
                getattr(self, f"alpha{i+1}") * H 
                for i, H in enumerate(H_hops)
            )
        
        # 线性变换 + 归一化 + 激活 + dropout
        H = self.lin(H_mix)
        H = self.ln(H)
        H = self.act(H)
        H = self.drop(H)
        return H


@register("model", "gcn_basicA_nhop")   
class GCNBasicNHop(nn.Module):
    """
    N-hop GCN model
    input:  batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
    output: pred_arr_reg (B,K), H_list (just for check)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",
                 out_array_dim=None, dropout=0.1, n_hops=2, alpha_init=1.0,out_scalar=False):
        """
        Args:
            in_dim: input dimension
            hidden: hidden dimension
            layers: number of GCN layers
            readout: readout method (mean/sum/max)
            out_array_dim: output array dimension
            dropout: dropout rate
            n_hops: number of hops (1, 2, 3, ..., n)
            alpha_init: initial alpha value
        """
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.gcns = nn.ModuleList([
            GraphConvNHop(dims[i], dims[i+1], dropout=dropout, 
                         n_hops=n_hops, alpha_init=alpha_init)
            for i in range(layers)
        ])
        self.readout = readout
        fuse_dim = hidden
        self.head_array = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None

    def _readout(self, H):
        if self.readout == "sum": return H.sum(1)
        if self.readout == "max": return H.max(1).values
        return H.mean(1)

    def forward(self, batch, return_all_H=True):
        A_hat, X = batch["A_hat"], batch["X"]
        H = X
        H_list = [H]
        for g in self.gcns:
            H = g(A_hat, H)      # (B,N,hidden)
            H_list.append(H)
        gcnout = self._readout(H)  # (B,hidden)

        out = {}
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
        if return_all_H:
            out["H_list"] = H_list
        return out


class APPNPConv(nn.Module):
    """
    APPNP propagation layer:
        1) H0 = f(X)  (一层线性 + ReLU)
        2) K 次迭代: H^{k+1} = (1 - alpha) A_hat H^k + alpha H0
    forward(A_hat, X)
        A_hat: [B, N, N]  (normalized adjacency with self-loops)
        X    : [B, N, F_in]
    """
    def __init__(self, in_dim, out_dim, K=10, alpha=0.1, dropout=0.0,
                 use_ln=True):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.use_ln = use_ln

        # f(X): 简单用一层线性 + ReLU（可以以后改成两层 MLP）
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

        self.ln = nn.LayerNorm(out_dim) if use_ln else None
        self.drop = nn.Dropout(dropout)

    def forward(self, A_hat, X):
        """
        A_hat: [B, N, N] or [N, N]
        X    : [B, N, F_in] or [N, F_in]
        """
        # 先把 A_hat / X 统一成 3D 形式，方便写 matmul
        if A_hat.dim() == 2:
            # [N,N] -> [1,N,N]
            A_hat = A_hat.unsqueeze(0)
        if X.dim() == 2:
            # [N,F] -> [1,N,F]
            X = X.unsqueeze(0)

        # H0 = f(X)
        H0 = self.lin(X)          # [B,N,out_dim]
        H0 = self.act(H0)

        # 迭代传播
        H = H0
        for _ in range(self.K):
            # [B,N,N] @ [B,N,F] -> [B,N,F]
            H = torch.matmul(A_hat, H)
            H = (1.0 - self.alpha) * H + self.alpha * H0

        if self.ln is not None:
            H = self.ln(H)
        H = self.drop(H)

        # 如果输入原来是 2D，就 squeeze 回去，保持接口自然
        if H.shape[0] == 1 and X.dim() == 2 + 0:  # 原始 X 是 2D
            H = H.squeeze(0)

        return H


@register("model", "mlp_basicA")
class MLPBasicA(nn.Module):
    """
    MLP-only baseline model (no graph structure).
    input : batch["A_hat"] (unused), batch["X"] (B,N,F)
    output: pred_arr_reg (B,K), H_list (for compatibility)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",
                 out_array_dim=None, dropout=0.1, out_scalar=False):
        super().__init__()
        self.readout = readout

        # 图级输入维度 = 节点特征 readout 之后的维度 = in_dim
        dims = [in_dim] + [hidden] * layers

        # 多层 MLP：Linear + ReLU + Dropout
        mlps = []
        for i in range(len(dims) - 1):
            mlps.append(nn.Linear(dims[i], dims[i+1]))
            mlps.append(nn.ReLU())
            if dropout > 0:
                mlps.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlps)

        fuse_dim = hidden
        self.head_array = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None

    def _readout(self, H):
        """
        H: [B, N, F]
        """
        if self.readout == "sum":
            return H.sum(1)          # [B, F]
        if self.readout == "max":
            return H.max(1).values   # [B, F]
        return H.mean(1)             # default: mean

    def forward(self, batch, return_all_H=True):
        # A_hat 不用，保持接口一致
        X = batch["X"]          # [B, N, F]

        # H_list 里至少放一个 X，方便你现有的 Gsmooth 代码使用
        H_list = [X]

        # 图级 pooled 特征：完全不看 A_hat
        g = self._readout(X)    # [B, in_dim]

        # 通过 MLP
        g = self.mlp(g)         # [B, hidden]

        out = {}
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(g)  # [B, out_array_dim]

        if return_all_H:
            out["H_list"] = H_list

        return out
 
@register("model", "gcn_basicA_appnp")
class GCNBasicAPPNP(nn.Module):
    """
    APPNP-style GCN model
    input : batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
    output: pred_arr_reg (B,K), H_list (for checking)
    """
    def __init__(self, in_dim, hidden=64, layers=1, readout="mean",
                 out_array_dim=None, dropout=0.1,
                 K=10, alpha=0.1, use_ln=True, out_scalar=False):
        super().__init__()

        # 和你之前一样的维度列表
        dims = [in_dim] + [hidden] * layers

        # 多层 APPNPConv 串联（通常 layers=1 就够了，你可以实验）
        self.gcns = nn.ModuleList([
            APPNPConv(dims[i], dims[i+1],
                      K=K, alpha=alpha,
                      dropout=dropout, use_ln=use_ln)
            for i in range(layers)
        ])

        self.readout = readout
        fuse_dim = hidden
        self.head_array = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None

    def _readout(self, H):
        if self.readout == "sum":
            return H.sum(1)
        if self.readout == "max":
            return H.max(1).values
        return H.mean(1)

    def forward(self, batch, return_all_H=True):
        A_hat, X = batch["A_hat"], batch["X"]   # [B,N,N], [B,N,F]
        H = X
        H_list = [H]
        for g in self.gcns:
            H = g(A_hat, H)          # (B,N,hidden)
            H_list.append(H)
        gcnout = self._readout(H)    # (B,hidden)

        out = {}
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
        if return_all_H:
            out["H_list"] = H_list
        return out




@register("model", "gcn_basicA_1_2hop")   
class GCNBasicOneTwoHop(nn.Module):
    """
      1-hop + 2-hop 
      input:  batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
      output: pred_arr_reg (B,K), H_list(just for check)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",
                 out_array_dim=None, dropout=0.1, out_scalar=False):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.gcns = nn.ModuleList([
            GraphConvOneTwoHop(dims[i], dims[i+1], dropout=dropout)
            for i in range(layers)
        ])
        self.readout = readout
        fuse_dim = hidden
        self.head_array = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None

    def _readout(self, H):
        if self.readout == "sum": return H.sum(1)
        if self.readout == "max": return H.max(1).values
        return H.mean(1)

    def forward(self, batch, return_all_H=True):
        A_hat, X = batch["A_hat"], batch["X"]
        H = X
        H_list = [H]
        for g in self.gcns:
            H = g(A_hat, H)      # (B,N,hidden)
            H_list.append(H)
        gcnout = self._readout(H)  # (B,hidden)

        out = {}
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
        if return_all_H:
            out["H_list"] = H_list
        return out


@register("model", "gcn_basicA")
class GCNBasic(nn.Module):
    """
      input:  batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
      output: pred_arr_reg (B,K)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",out_array_dim=None, dropout=0.1, out_scalar=False):
        super().__init__()
        dims = [in_dim] + [hidden]*layers
        self.gcns = nn.ModuleList([GraphConv(dims[i], dims[i+1], dropout) for i in range(layers)])
        self.readout = readout
        fuse_dim = hidden
        self.head_array  = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None 
        
    def _readout(self, H):
        if self.readout == "sum": return H.sum(1)
        if self.readout == "max": return H.max(1).values
        return H.mean(1)

    def forward(self, batch, return_all_H = True):
        A_hat, X = batch["A_hat"], batch["X"]
        H = X
        H_list = [H]
        for g in self.gcns:
            H = g(A_hat, H)  
            H_list.append(H)  # (B,N,hidden)
        gcnout = self._readout(H)              # (B,hidden)

        out = {}
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
            
        if return_all_H:
           out["H_list"] = H_list
           
        return out
    
    
@register("model", "gcn_basicA_1")
class GCNBasic(nn.Module):
    """
      input:  batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
      output: pred_arr_reg (B,K),logits(B,K)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",out_array_dim=None, dropout=0.1, out_scalar=False):
        super().__init__()
        dims = [in_dim] + [hidden]*layers
        self.gcns = nn.ModuleList([GraphConv(dims[i], dims[i+1], dropout) for i in range(layers)])
        self.readout = readout
        fuse_dim = hidden
        self.head_array  = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None 
        self.head_logits = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None 
    def _readout(self, H):
        if self.readout == "sum": return H.sum(1)
        if self.readout == "max": return H.max(1).values
        return H.mean(1)

    def forward(self, batch):
        A_hat, X = batch["A_hat"], batch["X"]
        H = X
        for g in self.gcns:
            H = g(A_hat, H)                  # (B,N,hidden)
        gcnout = self._readout(H)              # (B,hidden)

        out = {}
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
        if self.head_logits is not None:
            out["logits"] = self.head_logits(gcnout)
        return out
    
    

@register("model", "gcn_globalB")
class GCNGlobal(nn.Module):
    """
      input:  batch["A_hat"] (B,N,N), batch["X"] (B,N,F), batch["global_vec"](B, G)
      output:  pred_y_reg (B,), pred_arr_reg (B,K), logits(B,C)
    """
    def __init__(self, in_dim, hidden=64, layers=2, readout="mean",
                 out_scalar=True, out_array_dim=None, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden]*layers
        self.gcns = nn.ModuleList([GraphConv(dims[i], dims[i+1], dropout) for i in range(layers)])
        self.readout = readout
        fuse_dim = hidden + 18  # global_vec dim=18
        self.head_scalar = nn.Linear(fuse_dim, 1) if out_scalar else None
        self.head_array  = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None

    def _readout(self, H):
        if self.readout == "sum": return H.sum(1)
        if self.readout == "max": return H.max(1).values
        return H.mean(1)

    def forward(self, batch):
        A_hat, X = batch["A_hat"], batch["X"]
        H = X
        for g in self.gcns:
            H = g(A_hat, H)                  # (B,N,hidden)
        gcnout = self._readout(H)              # (B,hidden)
        g_vec = batch.get("global_vec", None)
        if g_vec is not None:
            gcnout = torch.cat([gcnout, g_vec], dim=-1)  # (B, hidden + G)
        out = {}
        if self.head_scalar is not None:
            out["pred_y_reg"] = self.head_scalar(gcnout).squeeze(-1)
        if self.head_array is not None:
            out["pred_arr_reg"] = self.head_array(gcnout)
        return out
