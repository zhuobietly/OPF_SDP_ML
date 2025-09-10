module ChordalPerturb
using PowerModels
export PerturbedGraph, generate_perturbations, generate_perturbations_from_adj
using SparseArrays, Random

"从 PowerModels 抽取对称 0/1(Float64) 邻接与 busID->index 映射"
function _adjacency_matrix(pm::AbstractPowerModel, nw::Int=nw_id_default)
    bus_ids  = ids(pm, nw, :bus)
    buspairs = ref(pm, nw, :buspairs)
    nb = length(bus_ids); nl = length(buspairs)
    lookup_index = Dict((bi, i) for (i, bi) in enumerate(bus_ids))
    f = [lookup_index[bp[1]] for bp in keys(buspairs)]
    t = [lookup_index[bp[2]] for bp in keys(buspairs)]
    A = sparse(vcat(f,t), vcat(t,f), ones(Float64, 2*nl), nb, nb)   # 对称 0/1 Float64
    return A, lookup_index
end

"一次候选图的信息载体"
struct PerturbedGraph
    adj::SparseMatrixCSC{Float64,Int}      # 对称 0/1 Float64
    lookup_index::Dict{Int,Int}            # busID -> 1..nb
    kind::Symbol                           # :original / :light / :heavy
    idx::Int                               # 第几次（原始=0）
    added_edges::Vector{Tuple{Int,Int}}    # (i<j)
end

# ---------------- 工具函数 ----------------

"规整为 0/1、对称、零对角；保持 Float64"
function _normalize_undirected(A::SparseMatrixCSC{<:Real,Int})
    nb = size(A,1)
    I,J,_ = findnz(A)
    B = sparse(I, J, ones(Float64, length(I)), nb, nb)  # 0/1
    C = max.(B, B')                                     # 并集确保对称
    for i in 1:nb
        if C[i,i] != 0.0
            C[i,i] = 0.0
        end
    end
    dropzeros!(C)
    return C
end

"“距离=2”的非边候选（内部临时用 Int 做稀疏乘法）"
function _distance2_candidates(A01::SparseMatrixCSC{<:Real,Int})
    nb = size(A01,1)
    I,J,_ = findnz(A01)
    Ai = sparse(I, J, ones(Int, length(I)), nb, nb)     # 0/1(Int)
    S  = Ai * Ai                                        # 计数两跳路径
    SI, SJ, _ = findnz(S)
    cands = Tuple{Int,Int}[]
    for k in eachindex(SI)
        i = SI[k]; j = SJ[k]
        if i < j && A01[i,j] == 0.0
            push!(cands, (i,j))
        end
    end
    return cands
end

"所有非边的兜底候选（规模大时会慢，仅用于补齐）"
function _all_nonedge_candidates(A01::SparseMatrixCSC{<:Real,Int})
    nb = size(A01,1)
    cands = Tuple{Int,Int}[]
    for j in 2:nb
        rows, _ = findnz(view(A01, :, j))   # 第 j 列非零行
        neigh = Set(rows)
        for i in 1:j-1
            if i ∉ neigh
                push!(cands, (i,j))
            end
        end
    end
    return cands
end

"不放回采样 m 条边"
function _sample_edges(cands::Vector{Tuple{Int,Int}}, m::Int, rng::AbstractRNG)
    m ≤ 0 && return Tuple{Int,Int}[]
    n = length(cands); m = min(m, n)
    n == 0 && return Tuple{Int,Int}[]
    idx = randperm(rng, n)
    return cands[idx[1:m]]
end

"在 A 上添加边集 E，返回对称 0/1(Float64)"
function _add_edges(A01::SparseMatrixCSC{<:Real,Int}, E::Vector{Tuple{Int,Int}})
    nb = size(A01,1)
    Ir, Jr, _ = findnz(A01)
    rows = Vector{Int}(Ir);  cols = Vector{Int}(Jr);  vals = ones(Float64, length(Ir))
    for (i,j) in E
        push!(rows, i); push!(cols, j); push!(vals, 1.0)
        push!(rows, j); push!(cols, i); push!(vals, 1.0)
    end
    return _normalize_undirected(sparse(rows, cols, vals, nb, nb))
end

# ---------------- 对外主函数 ----------------

"""
    generate_perturbations(pm, nw; light_k=6, heavy_k=3,
        light_edges=20, heavy_edges=50, dist2_only=true, rng=Random.default_rng())

返回长度 1+light_k+heavy_k 的 `Vector{PerturbedGraph}`：
- [1] 原始图 (:original, idx=0)
- [2..1+light_k] 轻扰动（每次加 `light_edges` 条）
- 其后 heavy_k 个为重扰动（每次加 `heavy_edges` 条）
"""
function generate_perturbations_original(pm::AbstractPowerModel, nw::Int;
    light_k::Int=6, heavy_k::Int=3,
    light_edges::Int=20, heavy_edges::Int=50,
    dist2_only::Bool=true, rng::AbstractRNG=Random.default_rng())
    adj, lookup_index = _adjacency_matrix(pm, nw)
    return generate_perturbations_from_adj(adj; lookup_index, light_k, heavy_k,
                                           light_edges, heavy_edges, dist2_only, rng)
end
function generate_perturbations(pm::AbstractPowerModel, nw::Int;
    light_frac::Float64 = 0.08,   # 轻扰动比例（默认 8%）
    heavy_frac::Float64 = 0.16,   # 重扰动比例（默认 16%）
    light_k::Int = 6, heavy_k::Int = 3,
    dist2_only::Bool = true,
    rng::AbstractRNG = Random.default_rng()
)
    # 原图邻接和索引
    adj, lookup_index = _adjacency_matrix(pm, nw)
    nb = size(adj, 1)
    m  = Int(nnz(adj) ÷ 2)                # 原始边数（无向）

    # 根据比例计算扰动边数
    light_edges = max(1, round(Int, light_frac * m))
    heavy_edges = max(light_edges, round(Int, heavy_frac * m))

    @info "Perturbation edges" nb=nb original_edges=m light_edges=light_edges heavy_edges=heavy_edges

    return generate_perturbations_from_adj(
        adj; lookup_index,
        light_k=light_k, heavy_k=heavy_k,
        light_edges=light_edges, heavy_edges=heavy_edges,
        dist2_only=dist2_only, rng=rng
    )
end


"无 PowerModels 依赖的版本：直接从邻接开跑（用于单元测试/小例子）"
function generate_perturbations_from_adj(adj::SparseMatrixCSC{<:Real,Int};
    lookup_index::Dict{Int,Int}=Dict(i=>i for i in 1:size(adj,1)),
    light_k::Int=6, heavy_k::Int=3,
    light_edges::Int=20, heavy_edges::Int=50,
    dist2_only::Bool=true, rng::AbstractRNG=Random.default_rng()
)
    A0 = _normalize_undirected(adj)
    out = PerturbedGraph[ PerturbedGraph(A0, lookup_index, :original, 0, Tuple{Int,Int}[]) ]

    d2_cands  = dist2_only ? _distance2_candidates(A0) : Tuple{Int,Int}[]
    all_cands = _all_nonedge_candidates(A0)

    # 轻扰动
    for k in 1:light_k
        take = _sample_edges(d2_cands, light_edges, rng)
        if length(take) < light_edges
            need = light_edges - length(take)
            used = Set(take)
            remain = [e for e in all_cands if !(e in used)]
            take  = vcat(take, _sample_edges(remain, need, rng))
        end
        push!(out, PerturbedGraph(_add_edges(A0, take), lookup_index, :light, k, take))
    end

    # 重扰动
    for k in 1:heavy_k
        take = _sample_edges(d2_cands, heavy_edges, rng)
        if length(take) < heavy_edges
            need = heavy_edges - length(take)
            used = Set(take)
            remain = [e for e in all_cands if !(e in used)]
            take  = vcat(take, _sample_edges(remain, need, rng))
        end
        push!(out, PerturbedGraph(_add_edges(A0, take), lookup_index, :heavy, k, take))
    end

    return out
end

end # module
