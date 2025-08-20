# ===== demo_fixed.jl =====
# 建议放在与 chordalvisual.jl 同目录
ENV["MPLBACKEND"] = "Agg"     # 远程/无GUI 时，避免 PyPlot 后端报错
using SparseArrays
using PyPlot
using PyCall

include("chordalvisual.jl")
import .ChordalVisualizer: visualize_fillin, edge_lists

# 小工具：从无向边集合构造对称邻接稀疏矩阵（自动做 i<j 去重，并补对称）
function adj_from_edges(n::Int, E::Vector{Tuple{Int,Int}})
    I = Int[]; J = Int[]
    seen = Set{Tuple{Int,Int}}()
    for (i,j) in E
        @assert 1 ≤ i ≤ n && 1 ≤ j ≤ n "edge ($(i),$(j)) out of range 1..$n"
        i == j && continue
        a, b = (i < j) ? (i,j) : (j,i)
        if !((a,b) in seen)
            push!(seen, (a,b))
            push!(I, a); push!(J, b)
            push!(I, b); push!(J, a)
        end
    end
    A = sparse(I, J, ones(Int, length(I)), n, n)
    for k in 1:n; A[k,k] = 0; end
    return max.(A, A')    # 对称化以防万一
end

const n_demo = 10

# 原始边（你想要的结构：一个10环 + 两条斜边）
const E_orig = [
    (1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,1),
    (2,6),(3,7)
]
const A_demo = adj_from_edges(n_demo, E_orig)

# 在原始边基础上，手动指定“填充边”
const E_fill = [
    (1,3),(2,4),(4,6),(5,7),(7,9),(1,9)
]
const C_demo = adj_from_edges(n_demo, vcat(E_orig, E_fill))

# 固定的重排（长度必须等于 n_demo）
const q_demo = [3, 1, 2, 5, 4, 6, 8, 10, 7, 9]

# 一键演示
function demo_fixed(; savepath::AbstractString = "result/figure/graph/fillin_demo_fixed.png",
                    use_perm::Bool = true)
    q = use_perm ? q_demo : nothing
    orig_edges, fill_edges = visualize_fillin(A_demo, C_demo; q=q, savepath=savepath)
    println("原始边数 = $(length(orig_edges)), 填充边数 = $(length(fill_edges))")
    return orig_edges, fill_edges
end

demo_fixed()

