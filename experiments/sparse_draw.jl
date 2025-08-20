using SparseArrays, PyPlot, Colors, PyCall

function visualize_fillin(original_adj::SparseMatrixCSC, cadj::SparseMatrixCSC; q=nothing, savepath="./result/figure/graph/fillin_grid.png")
    n = size(original_adj, 1)
    @assert size(cadj,1) == n "cadj must be same size as original_adj"

    # 应用重排序
    A = original_adj .!= 0
    C = cadj .!= 0
    if q !== nothing
        A = A[q, q]
        C = C[q, q]
    end

    # 构造颜色矩阵 M
    M = zeros(Int, n, n)
    for i in 1:n, j in 1:n
        if A[i,j]
            M[i,j] = 1      # 原始边
        elseif C[i,j]
            M[i,j] = 2      # fill-in 边
        end
    end

    # 颜色映射
    colors_array = ["white", "gray", "red"]
    matplotlib = pyimport("matplotlib")
    cmap = matplotlib.colors.ListedColormap(colors_array)

    # 画图
    figure(figsize=(5,5))
    imshow(M[end:-1:1, :], cmap=cmap, interpolation="none")  # 翻转行顺序使上三角在右上
    xticks(0:n-1, 1:n)
    yticks(0:n-1, n:-1:1)
    gca().set_xticks(collect(0.5:n), minor=true)
    gca().set_yticks(collect(0.5:n), minor=true)
    grid(which="minor", color="gray", linewidth=0.5)
    title("Matrix Sparsity: Original (gray), Fill-in (red)")
    tight_layout()
    savefig(savepath)
    println("✅ 网格图已保存为：$savepath")
end

function visualize_example_fillin_grid(savepath::String="./result/figure/graph/fillin_grid.png")
    # 原始图的邻接边
    orig_edges = [(1,2), (2,3), (3,4), (4,5), (2,5), (5,6), (6,7)]
    fillin_edges = [(2,4), (3,5)]
    n = 7

    # 构造原始邻接矩阵 A
    A = spzeros(Bool, n, n)
    for (i,j) in orig_edges
        A[i,j] = true
        A[j,i] = true
    end

    # 构造扩展邻接矩阵 C
    C = copy(A)
    for (i,j) in fillin_edges
        C[i,j] = true
        C[j,i] = true
    end

    # 生成填色矩阵 M:
    # 0: 空白
    # 1: 原始边
    # 2: fill-in 边
    M = zeros(Int, n, n)
    for i in 1:n, j in 1:n
        if A[i,j]
            M[i,j] = 1
        elseif C[i,j]
            M[i,j] = 2
        end
    end

    # 颜色映射：0白，1灰，2红
    colors_array = ["white", "gray", "red"]
  
    matplotlib = pyimport("matplotlib")
    cmap = matplotlib.colors.ListedColormap(colors_array)

    # 画图
    figure(figsize=(5,5))
    imshow(M[end:-1:1, :], cmap=cmap, interpolation="none")  # 翻转行顺序使下标向上
    xticks(0:n-1, 1:n)
    yticks(0:n-1, n:-1:1)
    gca().set_xticks(collect(0.5:n), minor=true)
    gca().set_yticks(collect(0.5:n), minor=true)
    grid(which="minor", color="gray", linewidth=0.5)
    title("Matrix Sparsity: Original (gray), Fill-in (red)")
    tight_layout()
    savefig(savepath)
    println("✅ 网格图已保存为：$savepath")
end

visualize_example_fillin_grid()
