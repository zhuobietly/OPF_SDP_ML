# memory_safe_load_solvegen.jl — cleaned with robust cleanup/finally blocks

import Pkg
# 若你的项目根目录并不是一个“注册包”，预编译失败不影响脚本运行；
# 为避免 Name = "Lanyue" 缺少源码时报错，可改为激活父目录但不强求依赖。
try
    Pkg.activate(joinpath(@__DIR__, ".."))   # activates the repo root as the project
    Pkg.instantiate()
catch
    @warn "Pkg.activate/instantiate failed; continuing with current environment"
end


# ---- local includes ----
include("../src_jl/solver_wrappers.jl")
include("../src_jl/LightGC.jl")    # 你需要在 src_jl/ 下新建 LightGC.jl，内容见上一条消息
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
using .LightGC

# ========== 可选：查看内存（主流程不调用） ==========
function print_memory_usage()
    println("🧠  Mem usage: ", round(Sys.total_memory() / 1024^3, digits=2), " GB total | ",
            round(Sys.free_memory() / 1024^3, digits=2), " GB free")
end

# 小工具：格式化 k
_fmt_k(k::Float64) = isnan(k) ? nothing : @sprintf("%.2f", k)

current_dir = @__DIR__
println("Current directory: ", current_dir)

# 基础案例与输入目录
case_file  = "case1888rte.m"
case_name  = replace(case_file, ".m" => "")
input_dir  = joinpath("/home/goatoine/Documents/Lanyue/data/load_profiles/", case_name)

# 三种 chordal formulation + 是否合并团
formulations = [Chordal_MD, Chordal_MFI, Chordal_AMD]
merging_opts = [true, false]
alpha_values = [3.0, 5.0]

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

    # 这些对象在 finally 里统一清理
    data = nothing
    loads = nothing
    df_scene = nothing

    try
        loads = JSON.parsefile(filepath)
        # 每个场景单独准备 data（避免相互污染）
        data_path = "/home/goatoine/Documents/Lanyue/data/raw_data/$case_file"
        data      = PowerModels.parse_file(data_path)

        # 轻量修改数据，不要额外复制大对象
        for (_gen_id, gen) in data["gen"]
            gen["cost"] .= gen["cost"] ./ 1e3
        end
        for (idd, load) in loads["load"]
            data["load"][idd]["pd"] = load["pd"]
            data["load"][idd]["qd"] = load["qd"]
        end

        # 从文件名解析：k/seed/id（仅使用 id）
        k_detect, seed_detect, id_detect = parse_k_seed_id_from_filename(json_file)
        k_token  = _fmt_k(k_detect)
        perturbation_name = replace(json_file, ".json" => "")
        load_id           = perturbation_name
        k_value = (k_token === nothing) ? NaN : parse(Float64, k_token)
        seed_value = (seed_detect === missing) ? NaN : Int(seed_detect)
        perturbation = (k_value, seed_value)

        # 该场景的"单独 CSV 文件名"
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

        df_scene = empty_results_df()
        CSV.write(results_csv, df_scene)  # 写空表头

        # 6 次求解（3 formulation × 2 merge）
        for fm in formulations
            for merging in merging_opts
                alpha_range = merging ? alpha_values : [0.0]  # 合并时用 alpha，否则 0.0
                for alpha in alpha_range
                    # 每次求解都用 result 局部变量，并在 finally 清理
                    result = nothing
                    try
                        println("Solving with formulation: ", fm, " | merging: ", merging, " | alpha: ", alpha)
                        save_name = "$(case_name)_$(fm)_$(merging)_$(isnothing(id_detect) ? "" : string(id_detect))"
                        result = SolverWrappers.solve(data, fm, merging, save_name;
                            alpha=alpha, id_name=json_file, tokens=tokens,
                            perturbation=perturbation, id_detect=id_detect)

                        solve_time  = get(result, "solve_time", NaN)
                        term_status = string(get(result, "termination_status", ""))
                        obj_val     = get(result, "objective", NaN)
                        sol_status  = string(get(result, "solution_status", ""))

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
                            ID              = [id_detect],
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
                            ID              = [id_detect],
                            load_id         = [load_id],
                        )
                        CSV.write(results_csv, row_err, append=true)
                    finally
                        # --- 每次求解后清理 ---
                        cleanup!(Ref(result))
                    end
                end
            end
        end
    finally
        # --- 每个文件循环末尾清理大对象 ---
        cleanup!(Ref(data), Ref(loads), Ref(df_scene))
    end
    print_memory_usage()
end

println("\n✅ 完成。每个场景已各自生成 1 个 CSV。")
