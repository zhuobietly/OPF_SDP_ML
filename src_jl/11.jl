using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using CSV
using DataFrames
using Random
using JSON
using Dates
using Printf

# 可选：查看内存（主流程不调用）
function print_memory_usage()
    println("🧠  Mem usage: ", round(Sys.total_memory() / 1024^3, digits=2), " GB total | ",
            round(Sys.free_memory() / 1024^3, digits=2), " GB free")
end

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

_fmt_k(k::Float64) = isnan(k) ? nothing : @sprintf("%.2f", k)

"""
构建 chordal-SDP 的 OPF 并用 MOSEK 求解。
（不计算/保存原始邻接，不调用 GC）
"""
function solve(data, model, clique_merging)
    pm = InfrastructureModels.InitializeInfrastructureModel(
        model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym
    )
    PowerModels.ref_add_core!(pm.ref)

    nw = collect(InfrastructureModels.nw_ids(pm, pm_it_sym))[1]
    println("Beginning chordal extension (merge = $(clique_merging))")

    cadj_chordal, lookup_index, sigma =
        PowerModels._chordal_extension(pm, nw; clique_merge=clique_merging)

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
case_file  = "case118.m"
case_name  = replace(case_file, ".m" => "")
input_dir  = joinpath(current_dir, "output", "PL_118_10")

# 配置
formulations = [Chordal_AMD, Chordal_MFI, Chordal_MD]
merging_opts = [true, false]

# 结果表列名（与你样例一致）
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

run_id_global = 0  # 全局递增 ID（保留）

for json_file in readdir(input_dir)
    endswith(json_file, ".json") || continue

    filepath = joinpath(input_dir, json_file)
    println("\nReading scenario: ", filepath)
    loads = JSON.parsefile(filepath)

    # 每个场景单独准备 data
    data_path = joinpath(current_dir, "data", case_file)
    data      = PowerModels.parse_file(data_path)
    for (id, load) in loads["load"]
        data["load"][id]["pd"] = load["pd"]
        data["load"][id]["qd"] = load["qd"]
    end

    # 从文件名解析：k/seed/id（仅使用 id）
    k_detect, seed_detect, id_detect = parse_k_seed_id_from_filename(json_file)
    k_token  = _fmt_k(k_detect)                 # "0.30" 或 nothing
    A_param  = isnan(k_detect) ? NaN : k_detect # A_parameter 列
    load_id  = (id_detect === missing) ? "" : string(id_detect)  # 仅用 id（=2）

    # 列 perturbation：保留文件名去后缀
    perturbation_name = replace(json_file, ".json" => "")

    # 每场景一个 CSV：仅用 id 命名，不带 seed
    # pglib_opf_<case>_k_<k>_perturbation_<id>.csv
    name_tokens = ["pglib_opf", case_name]
    if k_token !== nothing
        append!(name_tokens, ["k", k_token])
    end
    append!(name_tokens, ["perturbation", (load_id == "" ? perturbation_name : load_id) * ".csv"])
    results_csv = joinpath(current_dir, join(name_tokens, "_"))
    println("Scene CSV -> ", results_csv)

    # 写表头
    CSV.write(results_csv, empty_results_df())

    # 3 × 2 = 6 次求解
    for fm in formulations
        for merging in merging_opts
            run_id_global += 1
            try
                println("Solving with formulation: ", fm, " | merging: ", merging)
                result = solve(data, fm, merging)

                solve_time  = get(result, "solve_time", NaN)
                term_status = string(get(result, "termination_status", ""))
                obj_val     = get(result, "objective", NaN)
                sol_status  = string(get(result, "solution_status", ""))  # 可能不存在

                row = DataFrame(
                    Formulation     = [string(fm)],
                    perturbation    = [perturbation_name],
                    Case            = [case_name],
                    Merge           = [merging],
                    A_parameter     = [A_param],
                    SolveTime       = [solve_time],
                    Status          = [term_status],
                    objective       = [obj_val],
                    SolutionStatus  = [sol_status],
                    ID              = [run_id_global],
                    load_id         = [load_id],   # 只用 id（=2）
                )
                CSV.write(results_csv, row, append=true)

            catch e
                @warn "Error solving with formulation=$(fm), merging=$(merging)" e
                row_err = DataFrame(
                    Formulation     = [string(fm)],
                    perturbation    = [perturbation_name],
                    Case            = [case_name],
                    Merge           = [merging],
                    A_parameter     = [A_param],
                    SolveTime       = [NaN],
                    Status          = ["error"],
                    objective       = [NaN],
                    SolutionStatus  = [""],
                    ID              = [run_id_global],
                    load_id         = [load_id],
                )
                CSV.write(results_csv, row_err, append=true)
            end
        end
    end
end

println("\n✅ 完成。每个场景已各自生成 1 个 CSV（命名只用 id）。")
