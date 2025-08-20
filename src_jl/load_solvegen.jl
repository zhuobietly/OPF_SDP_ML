import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activates the repo root as the project
Pkg.instantiate()
include("../src_jl/solver_wrappers.jl")
using PowerModels
Base.eval(PowerModels, :(Memento.setlevel!(Memento.getlogger(PowerModels), "error")))
using InfrastructureModels
using Mosek
using MosekTools
using CSV
using DataFrames
using Random
using JSON
using Dates
using Printf
using .SolverWrappers
# ========== 可选：查看内存（主流程不调用） ==========
function print_memory_usage()
    println("🧠  Mem usage: ", round(Sys.total_memory() / 1024^3, digits=2), " GB total | ",
            round(Sys.free_memory() / 1024^3, digits=2), " GB free")
end

# 从路径里尽力解析 k 与 seed：匹配 "_0.03_perturbation_301_2" 这类片段
function infer_k_seed(strs::Vector{String})
    for s in strs
        m = match(r"_([0-9\.]+)_perturbation_([0-1000]+)_([0-9]+)", s)
        if m !== nothing
            k  = try parse(Float64, m.captures[1]) catch; NaN; end
            id = try parse(Int,     m.captures[2]) catch; -1;  end
            return (k, id)
        end
    end
    return (NaN, -1)
end

# 小工具：格式化 k
_fmt_k(k::Float64) = isnan(k) ? nothing : @sprintf("%.2f", k)

"""
solve(data, model, clique_merging)
构建 chordal-SDP 的 OPF 并用 MOSEK 求解。
（不再计算/保存原始邻接，不调用 GC）
"""
function solve(data, model, clique_merging)
    pm = InfrastructureModels.InitializeInfrastructureModel(
        model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym
    )
    PowerModels.ref_add_core!(pm.ref)

    nw = collect(InfrastructureModels.nw_ids(pm, pm_it_sym))[1]
    println("Beginning chordal extension (merge = $(clique_merging))")

    # 仅做弦图扩展 + 团分解
    cadj_chordal, lookup_index, sigma = PowerModels._chordal_extension(pm, nw, clique_merging)

    cliques = PowerModels._maximal_cliques(cadj_chordal)
    lookup_bus_index = Dict((reverse(p) for p in pairs(lookup_index)))
    groups = [[lookup_bus_index[gi] for gi in g] for g in cliques]

    pm.ext[:SDconstraintDecomposition] =
        PowerModels._SDconstraintDecomposition(groups, lookup_index, sigma)

    println("Building the OPF")
    PowerModels.build_opf(pm)

    result = optimize_model!(pm, optimizer=Mosek.Optimizer)
    return result
end

# ================= 主脚本 =================
current_dir = @__DIR__
println("Current directory: ", current_dir)

# 基础案例与输入目录
case_file  = "case2746wop.m"      
case_name  = replace(case_file, ".m" => "")
input_dir  = joinpath("/home/goatoine/Documents/Lanyue/data/load_profiles/", case_name)

# 三种 chordal formulation + 是否合并团
formulations = [Chordal_MFI, Chordal_AMD, Chordal_MD]
merging_opts = [true, false]
alpha_values = [3.0, 4.0, 5.0]
# 解析文件名：case14_0.30_perturbation_301_2.json
# 返回 (k, seed, id)
function parse_k_seed_id_from_filename(fname::String)
    m = match(r"^[A-Za-z0-9]+_([0-9]+(?:\.[0-9]+)?)_perturbation_([0-9]+)_([0-9]+)\.json$", fname)
    if m === nothing
        return (NaN, missing, missing)
    end
    k     = try parse(Float64, m.captures[1]) catch; NaN; end
    seed  = try parse(Int,     m.captures[2]) catch; missing; end   # 301（不用）
    idno  = try parse(Int,     m.captures[3]) catch; missing; end   # 2（要用）
    return (k, seed, idno)
end


# 结果表的列名（与你的样例完全一致）
function empty_results_df()
    return DataFrame(
        Formulation     = String[],
        perturbation    = String[],
        Case            = String[],
        Merge           = Bool[],
        A_parameter     = Float64[],
        SolveTime       = Float64[],
        Status          = String[],
        objective       = Float64[],
        SolutionStatus  = String[],
        ID              = Int[],
        load_id         = String[],
    )
end


for json_file in readdir(input_dir)
    endswith(json_file, ".json") || continue

    filepath = joinpath(input_dir, json_file)
    println("\nReading scenario: ", filepath)
    loads = JSON.parsefile(filepath)

    # 每个场景单独准备 data（避免相互污染）
    data_path = "/home/goatoine/Documents/Lanyue/data/raw_data/$case_file"
    data      = PowerModels.parse_file(data_path)
    for (_gen_id, gen) in data["gen"]
        println("gen cost before: ", gen["cost"])
        gen["cost"] .= gen["cost"] ./ 1e3
        println("gen cost after: ", gen["cost"])
    end
    for (idd, load) in loads["load"]
        data["load"][idd]["pd"] = load["pd"]
        data["load"][idd]["qd"] = load["qd"]
    end
    # 从文件名解析：k/seed/id（仅使用 id）
    k_detect, seed_detect, id_detect = parse_k_seed_id_from_filename(json_file)
    k_token  = _fmt_k(k_detect)  
    seed_token = isnan(seed_detect) ? NaN : seed_detect # "0.30" 或 nothing
    id  = (id_detect === missing) ? "" : string(id_detect)  # 仅用 id（=2）
    # 场景标识
    perturbation_name = replace(json_file, ".json" => "")
    load_id           = perturbation_name
    # A_parameter（k）：若能识别写 k，否则 NaN
    k_value = (k_token === nothing) ? NaN : parse(Float64, k_token)
    seed_value = (seed_detect === missing) ? NaN : Int(seed_detect)
    perturbation = (k_value, seed_value)
    
    # 该场景的"单独 CSV 文件名"
    # 格式：pglib_opf_<case>_k_<k>_<id>_perturbation.csv
    tokens = ["pglib_opf", case_name]
    if k_token !== nothing && !ismissing(id_detect)
        append!(tokens, ["k", k_token, string(id_detect), "perturbation.csv"])
    else
        append!(tokens, ["perturbation.csv"])
    end
    csv_path = "/home/goatoine/Documents/Lanyue/data/solve_time/$case_name"
    !isdir(csv_path) && mkpath(csv_path)
    results_csv = joinpath(csv_path, join(tokens, "_"))
    println("Scene CSV -> ", results_csv)

    # 为该场景新建一个空 DataFrame，并先写表头
    df_scene = empty_results_df()
    CSV.write(results_csv, df_scene)  # 写空表头

    # 6 次求解（3 formulation × 2 merge）
    for fm in formulations
        for merging in merging_opts
            alpha_range = merging ? alpha_values : [0.0]  # 合并时用 alpha，否则 NaN
            for alpha in alpha_range
                try
                    println("Solving with formulation: ", fm, " | merging: ", merging, " | alpha: ", alpha)
                    save_name = "$(case_name)_$(fm)_$(merging)_$(id)"
                    result = SolverWrappers.solve(data, fm, merging, save_name; alpha=alpha)
                    solve_time  = get(result, "solve_time", NaN)
                    term_status = string(get(result, "termination_status", ""))
                    obj_val     = get(result, "objective", NaN)
                    sol_status  = string(get(result, "solution_status", ""))  # 可能不存在

                    row = DataFrame(
                        Formulation     = [string(fm)],
                        perturbation    = [perturbation],
                        Case            = [case_name],
                        Merge           = [merging],
                        A_parameter     = [alpha],
                        SolveTime       = [solve_time],
                        Status          = [term_status],
                        objective       = [obj_val],
                        SolutionStatus  = [sol_status],
                        ID              = [id_detect],  # 使用 id_detect
                        load_id         = [load_id],
                    )
                    CSV.write(results_csv, row, append=true)

                catch e
                    @warn "Error solving with formulation=$(fm), merging=$(merging), alpha=$(alpha)" e
                    row_err = DataFrame(
                        Formulation     = [string(fm)],
                        perturbation    = [perturbation],
                        Case            = [case_name],
                        Merge           = [merging],
                        A_parameter     = [alpha],
                        SolveTime       = [NaN],
                        Status          = ["error"],
                        objective       = [NaN],
                        SolutionStatus  = [""],
                        ID              = [id_detect],  # 使用 id_detect
                        load_id         = [load_id],
                    )
                    CSV.write(results_csv, row_err, append=true)
                end
            end
        end
    end
end

println("\n✅ 完成。每个场景已各自生成 1 个 CSV。")
