module ChordalVisualizer

using SparseArrays
using PyPlot
using PyCall
export visualize_fillin, edge_lists, edge_lists3, visualize_fillin3

"把稀疏邻接矩阵转成无向边列表（只保留 i<j，一条边一份）"
function undirected_edge_list(M::SparseMatrixCSC)
    I, J, _ = findnz(M)
    s = Set{Tuple{Int,Int}}()
    @inbounds for k in eachindex(I)
        i = I[k]; j = J[k]
        i == j && continue                 # 去掉对角
        a = i < j ? i : j                  # 无向化（端点排序）
        b = i < j ? j : i
        push!(s, (a,b))
    end
    return sort!(collect(s))               # Vector{Tuple{Int,Int}}，元素 (i,j) 且 i<j
end

"""
从 original_adj、cadj 得到
- original_edges：原始边
- fillin_edges：cadj 中新增的边（相对 original）
可选 q：若给定（1:n 的置换），先对两个矩阵做相同重排再提取（便于与可视化对齐）。
"""
function edge_lists(original_adj::SparseMatrixCSC, cadj::SparseMatrixCSC; q=nothing)
    n = size(original_adj,1)
    @assert size(original_adj,1)==size(original_adj,2) "original_adj must be square"
    @assert size(cadj,1)==size(cadj,2)==n "cadj must be same size as original_adj"

    A = original_adj .!= 0
    C = cadj         .!= 0
    # 去对角
    @inbounds for i in 1:n
        A[i,i] = false
        C[i,i] = false
    end

    if q !== nothing
        @assert length(q)==n "q must be a permutation of 1:n"
        A = A[q,q]
        C = C[q,q]
    end

    orig = undirected_edge_list(A)
    comp = undirected_edge_list(C)

    # 计算 fill-in = C \ A
    orig_set = Set(orig)
    fillin = [e for e in comp if !(e in orig_set)]
    return orig, fillin
end

"""
    visualize_fillin(original_adj, cadj; q=nothing, savepath="fillin_grid.png")

显示原始邻接与三角化后（含 fill-in）的差异网格：
灰色=原始边；红色=fill-in 边（cadj 有但 original 没有）。
- `q`：1:n 的置换，仅用于显示重排与边列表提取顺序；不做子集。
- 会自动 mkpath 保存目录。
- 返回 `(original_edges, fillin_edges)`（元素为 `(i,j)`，且 `i<j`）。
"""
function visualize_fillin(original_adj::SparseMatrixCSC, cadj::SparseMatrixCSC;
    q=nothing, savepath::AbstractString="fillin_grid.png",
    # ↓ 新增参数
    px_per_cell::Real=6,         # 每个单元格希望的像素数
    dpi::Integer=120,            # 图像分辨率
    max_inches::Real=18,         # 最大边长（英寸）
    show_ticks::Symbol=:auto,    # :auto / :on / :off
    show_grid::Symbol=:auto      # :auto / :on / :off
)
    n = size(original_adj,1)
    @assert size(original_adj,1)==size(original_adj,2)
    @assert size(cadj,1)==size(cadj,2)==n

    # —— 计算画布尺寸（英寸）= 像素 / dpi ——
    w_in = clamp(n*px_per_cell/dpi, 4, max_inches)
    h_in = w_in

    # ……(边列表、A/C、M 的构造同原来，不再赘述)……
    A = original_adj .!= 0; C = cadj .!= 0
    for i in 1:n; A[i,i]=false; C[i,i]=false; end
    if q !== nothing; A=A[q,q]; C=C[q,q]; end
    M = zeros(Int,n,n)
    @inbounds for i in 1:n, j in 1:n
        M[i,j] = A[i,j] ? 1 : (C[i,j] ? 2 : 0)
    end

    matplotlib = pyimport("matplotlib")
    cmap = matplotlib.colors.ListedColormap(["white","gray","red"])

    # 用计算得到的尺寸与 dpi
    figure(figsize=(w_in,h_in), dpi=dpi)
    imshow(M[end:-1:1, :], cmap=cmap, interpolation="none", vmin=0, vmax=2)

    # —— 轴刻度与网格的自适应 ——
    # 只显示 ~20 个刻度
    step = max(1, ceil(Int, n/20))
    if show_ticks == :off || (show_ticks == :auto && n > 120)
        xticks([]); yticks([])
    else
        xticks(0:step:n-1, 1:step:n)
        yticks(0:step:n-1, n:-step:1)
        gca().tick_params(axis="x", labelrotation=90)
    end

    if show_grid == :on || (show_grid == :auto && n <= 120)
        ax = gca()
        ax.set_xticks(collect(0.5:1:n-0.5), minor=true)
        ax.set_yticks(collect(0.5:1:n-0.5), minor=true)
        grid(which="minor", linewidth=0.5)
    end

    title("Matrix Sparsity: Original (gray), Fill-in (red)")
    tight_layout()
    try; mkpath(dirname(savepath)); catch; end
    savefig(savepath, bbox_inches="tight")
    close()
    println("✅ 保存：$savepath  | 尺寸：$(round(w_in,digits=2))\" × $(round(h_in,digits=2))\" @ $(dpi) dpi")
end


# === 新增：三类边列表（原始 / 扰动新增 / 填充） ===
"""
    edge_lists3(A0, A1, C; q=nothing)

给定：
- `A0`：原始邻接（PowerModels 原始网络）
- `A1`：扰动后的邻接（在 `A0` 上通过 network_perturbation 添加蓝色边得到）
- `C` ：三角化/弦化后的邻接（`cadj`）

返回 3 个无向边列表（元素均为 `(i,j)` 且 i<j）：
`(original_edges, added_by_perturb, fillin_edges)`。

若给定 `q`（1:n 的置换），会对三个矩阵做同一重排，使结果与显示一致。
"""
function edge_lists3(A0::SparseMatrixCSC, A1::SparseMatrixCSC, C::SparseMatrixCSC; q=nothing)
    n = size(A0,1)
    @assert size(A0,2)==n && size(A1,1)==n && size(A1,2)==n && size(C,1)==n && size(C,2)==n

    B0 = A0 .!= 0
    B1 = A1 .!= 0
    BC = C  .!= 0
    for i in 1:n
        B0[i,i] = false; B1[i,i] = false; BC[i,i] = false
    end
    if q !== nothing
        @assert length(q)==n
        B0 = B0[q,q]; B1 = B1[q,q]; BC = BC[q,q]
    end

    # 分类：原始=灰；扰动新增=蓝（A1 \ A0）；fill-in=红（C \ A1）
    orig  = undirected_edge_list(B0)
    added = undirected_edge_list(B1 .& .!B0)
    fill  = undirected_edge_list(BC .& .!B1)
    return orig, added, fill
end

# === 新增：三色网格图（灰=原始, 蓝=扰动新增, 红=fill-in） ===
"""
    visualize_fillin3(A0, A1, C; q=nothing, savepath="fillin3.png", ...)

渲染三类边的稀疏模式：
- 灰色：原始边（A0）
- 蓝色：扰动新增边（A1 \\ A0）
- 红色：弦化填充边（C \\ A1）

返回 `(original_edges, added_by_perturb, fillin_edges)`。
"""
function visualize_fillin3(A0::SparseMatrixCSC, A1::SparseMatrixCSC, C::SparseMatrixCSC;
    q=nothing, savepath::AbstractString="fillin3.png",
    px_per_cell::Real=6, dpi::Integer=120, max_inches::Real=18,
    show_ticks::Symbol=:auto, show_grid::Symbol=:auto
)
    n = size(A0,1)
    @assert size(A0,2)==n && size(A1,1)==n && size(A1,2)==n && size(C,1)==n && size(C,2)==n

    # 三类布尔矩阵
    B0 = A0 .!= 0;  B1 = A1 .!= 0;  BC = C .!= 0
    for i in 1:n
        B0[i,i]=false; B1[i,i]=false; BC[i,i]=false
    end
    if q !== nothing
        @assert length(q)==n
        B0 = B0[q,q]; B1 = B1[q,q]; BC = BC[q,q]
    end

    # 0=空, 1=原始灰, 2=扰动蓝, 3=fill-in红
    M = zeros(Int, n, n)
    @inbounds for i in 1:n, j in 1:n
        if B0[i,j]
            M[i,j] = 1
        elseif B1[i,j]    # 在 A1 但不在 A0
            M[i,j] = 2
        elseif BC[i,j]    # 在 C 但不在 A1
            M[i,j] = 3
        else
            M[i,j] = 0
        end
    end

    # 画图参数
    w_in = clamp(n*px_per_cell/dpi, 4, max_inches)
    h_in = w_in

    matplotlib = pyimport("matplotlib")
    # 白 / 灰 / 蓝 / 红
    cmap = matplotlib.colors.ListedColormap(["white", "#b0b0b0", "#1f77b4", "white"])

    figure(figsize=(w_in,h_in), dpi=dpi)
    imshow(M[end:-1:1, :], cmap=cmap, interpolation="none", vmin=0, vmax=3)

    # 轴刻度/网格与原函数一致
    step = max(1, ceil(Int, n/20))
    if show_ticks == :off || (show_ticks == :auto && n > 120)
        xticks([]); yticks([])
    else
        xticks(0:step:n-1, 1:step:n)
        yticks(0:step:n-1, n:-step:1)
        gca().tick_params(axis="x", labelrotation=90)
    end
    if show_grid == :on || (show_grid == :auto && n <= 120)
        ax = gca()
        ax.set_xticks(collect(0.5:1:n-0.5), minor=true)
        ax.set_yticks(collect(0.5:1:n-0.5), minor=true)
        grid(which="minor", linewidth=0.5)
    end

    title("Matrix Sparsity: Original(gray), Perturbation(blue), Fill-in(red)")
    tight_layout()
    try; mkpath(dirname(savepath)); catch; end
    savefig(savepath, bbox_inches="tight"); close()

    # 返回三类边列表，便于你外面统计/验证
    return edge_lists3(A0, A1, C; q=q)
end



end # module
