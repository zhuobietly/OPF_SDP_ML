module LoadProfileGen
using Random
using JSON
using Printf
using PowerModels
export create_load_profiles_from_matpower, create_load_profiles_from_json, create_load_profiles_from_case

# =============== 工具函数 =============== #

"安全写 JSON（一次性完整写入）。"
function _write_json(path::AbstractString, obj)
    open(path, "w") do io
        JSON.print(io, obj, 4)
    end
end

"将 sigmas 参数统一为 Float64 数组。"
function _normalize_sigmas(sigmas)::Vector{Float64}
    if sigmas isa Number
        return [Float64(sigmas)]
    else
        arr = Float64.(collect(sigmas))
        isempty(arr) && error("sigmas 不能为空")
        return arr
    end
end

"读取目录中现有 JSON 的最大编号，返回“下一个编号”（兼容纯缩放文件通过内容里的 index 推断）。"
function _next_global_id(dir::AbstractString)::Int
    isdir(dir) || return 1
    maxid = 0
    for f in readdir(dir; join=true)
        endswith(f, ".json") || continue
        name = basename(f)
        # 优先从文件名尾部的 _<id>.json 提取
        m = match(r"_(\d+)\.json$", name)
        if m !== nothing
            id = parse(Int, m.captures[1])
            maxid = max(maxid, id)
        else
            # 例如 pure_p_XX.json：从内容里读某一条 load 的 index
            try
                obj = JSON.parsefile(f)
                if haskey(obj, "load")
                    any_entry = first(values(obj["load"]))
                    if haskey(any_entry, "index")
                        id = Int(any_entry["index"])
                        maxid = max(maxid, id)
                    end
                end
            catch
                # 忽略不可解析文件
            end
        end
    end
    return maxid + 1
end

# =============== 主函数 =============== #

"""
    create_load_profiles_from_case(
        case_tag::AbstractString,
        data::Dict{String,Any},
        N::Int,
        outputDIR::AbstractString;
        σ_max::Vector{<:Real}=[0.05,0.06,0.07],  # 修改这里
        emit_pure::Bool=false,
        sigmas::Union{Real,AbstractVector{<:Real}}=σ_max,
        couple_pq::Bool=false,
        coeffs::AbstractVector{<:Real}=collect(0.8:0.05:1.2),
        start_index::Union{Nothing,Int}=nothing
    ) -> Nothing

批量生成负荷场景；每个输出文件内的所有负荷条目 `"index"` 都等于该文件的唯一编号（从 1 开始连续递增）。
"""
function create_load_profiles_from_case(
    case_tag::AbstractString,
    data::Dict{String,Any},
    N::Int,
    outputDIR::AbstractString;
    σ_max::Vector{<:Real}=[0.07],  # 修改这里
    emit_pure::Bool=false,
    sigmas::Union{Real,AbstractVector{<:Real}}=σ_max,
    couple_pq::Bool=false,
    coeffs::AbstractVector{<:Real}=collect(0.8:0.2:1.2),
    start_index::Union{Nothing,Int}=nothing
)::Nothing

    mkpath(outputDIR)
    @assert haskey(data, "load") "输入 data 必须包含键 \"load\""

    original_loads = deepcopy(data["load"])  # Dict{String, Dict{String,Any}}
    sigma_list = _normalize_sigmas(sigmas)
    coeffs64 = Float64.(collect(coeffs))

    # 估算总文件数，用于说明；也可以不依赖这个值，下面用 file_id 逐个增长
    total_files = (emit_pure ? length(coeffs64) : 0) + length(coeffs64) * length(sigma_list) * N

    # 全局唯一文件编号：从目录中续接或使用显式起点
    file_id = isnothing(start_index) ? _next_global_id(outputDIR) : start_index

    # 遍历比例系数
    for (i, coeff) in enumerate(coeffs64)
        # 生成基线负荷（按比例缩放）
        base_loads = Dict{String,Dict{String,Any}}()
        for (id, load) in original_loads
            pd0 = haskey(load, "pd") ? Float64(load["pd"]) : 0.0
            qd0 = haskey(load, "qd") ? Float64(load["qd"]) : 0.0
            bus = Int(get(load, "bus", get(load, "load_bus", -1)))
            status = Int(get(load, "status", 1))
            idx = try
                parse(Int, id)
            catch
                get(load, "index", -1)
            end
            base_loads[id] = Dict(
                "pd" => coeff * pd0,
                "qd" => coeff * qd0,
                "load_bus" => bus,
                "status" => status,
                "index" => file_id,                 # 先占位；真正写文件时会统一覆盖为 file_id
                "source_id" => Any["bus", bus],
            )
        end

        # 可选：导出纯缩放样本（覆盖 index = file_id）
        if emit_pure
            pure_loads = Dict{String,Dict{String,Any}}()
            for (id, bload) in base_loads
                pure_loads[id] = Dict(
                    "source_id" => bload["source_id"],
                    "load_bus"  => bload["load_bus"],
                    "status"    => bload["status"],
                    "index"     => file_id,      # 统一写成当前文件的唯一编号
                    "pd"        => bload["pd"],
                    "qd"        => bload["qd"],
                )
            end
            out_pure = Dict("load" => pure_loads)
            pure_name = @sprintf("pure_p_%02d.json", i)
            _write_json(joinpath(outputDIR, pure_name), out_pure)
            file_id += 1
        end

        # 在基线上叠加噪声（覆盖 index = file_id）
        for (k, σ) in enumerate(sigma_list)
            for j in 1:N
                # 为 (i, k, j) 生成唯一可复现实的种子
                seed = (i-1)*N*length(sigma_list) + (j-1) + round(Int, σ*1000)
                Random.seed!(seed)

                new_loads = Dict{String,Dict{String,Any}}()
                for (id, bload) in base_loads
                    pd_base = bload["pd"]
                    qd_base = bload["qd"]

                    if couple_pq
                        R = randn()
                        δp = R * abs(σ * pd_base)
                        δq = R * abs(σ * qd_base)
                    else
                        δp = randn() * abs(σ * pd_base)
                        δq = randn() * abs(σ * qd_base)
                    end

                    new_pd = pd_base + δp
                    new_qd = qd_base + δq

                    # 对原本为正的负荷进行非负截断
                    if pd_base > 0
                        new_pd = max(0.0, new_pd)
                    end
                    if qd_base > 0
                        new_qd = max(0.0, new_qd)
                    end

                    new_loads[id] = Dict(
                        "source_id" => bload["source_id"],
                        "load_bus"  => bload["load_bus"],
                        "status"    => bload["status"],
                        "index"     => file_id,    # 统一写成当前文件的唯一编号
                        "pd"        => new_pd,
                        "qd"        => new_qd,
                    )
                end

                out_dict = Dict("load" => new_loads)
                σstr = @sprintf("%.2f", σ)
                fname = @sprintf("%s_%s_perturbation_%d_%d.json", case_tag, σstr, seed, file_id)

                _write_json(joinpath(outputDIR, fname), out_dict)
                file_id += 1
            end
        end
    end

    println("✅ 已生成 $total_files 个文件；最后一个文件编号为 $(file_id-1)。")
    return nothing
end

# =============== 从 JSON 文件起步 =============== #

"""
    create_load_profiles_from_json(json_path::AbstractString, N, outputDIR; kwargs...)

从形如 {\"load\": {...}} 的 JSON 文件起步，解析后调用 `create_load_profiles_from_case`。
"""
function create_load_profiles_from_json(
    json_path::AbstractString,
    N::Int,
    outputDIR::AbstractString;
    kwargs...
)::Nothing
    data = JSON.parsefile(json_path)
    case_tag = splitext(basename(json_path))[1]
    create_load_profiles_from_case(case_tag, data, N, outputDIR; kwargs...)
    return nothing
end

# =============== 从 MATPOWER 算例起步（需要 PowerModels） =============== #

"""
    create_load_profiles_from_matpower(case_path::AbstractString, N, outputDIR; kwargs...)

从 MATPOWER/PGLib 的 .m 算例读取，随后生成负荷场景。
"""
function create_load_profiles_from_matpower(
    case_path::AbstractString,
    N::Int,
    outputDIR::AbstractString;
    kwargs...
)::Nothing
    data = PowerModels.parse_file(case_path)
    case_tag = splitext(basename(case_path))[1]
    create_load_profiles_from_case(case_tag, data, N, outputDIR; kwargs...)
    return nothing
end

end # module
