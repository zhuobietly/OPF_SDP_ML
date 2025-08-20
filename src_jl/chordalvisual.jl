module ChordalVisualizer

using SparseArrays
using PyPlot
using PyCall
export visualize_fillin, edge_lists

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



end # module
