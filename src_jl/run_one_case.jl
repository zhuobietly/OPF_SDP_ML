import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("../src_jl/solver_wrappers.jl")
# ---- Tunables (可用 ENV 覆盖) ----
const EST_KAPPA         = 8.0    # 原 9.0
const MEM_BUDGET_FRAC   = 0.95   # 原 0.92
const MEM_BUDGET_RES_GB = 0.5    # 原 1.0
const MEM_BUDGET_MIN_GB = 3.0    # 保持
const EST_TOL           = 1.25   # 原 1.15

using CSV, DataFrames

# 与现有 CSV 一致的列顺序（按你的表头）
const EXPECTED_ORDER = [
    "Formulation","Perturbation","Case","Merge","A_parameter","SolveTime",
    "Status","objective","SolutionStatus","ID","load_id",
    "Iterations","PrimalRes","DualRes","RelGap","KKTCondProxy","ActiveLimits",
    "r_max","t","sum_r_sq","sum_r_cu","sep_max","sep_mean","sum_sep_sq",
    "tree_max_deg","tree_h","fillin","coupling"
]

# 追加一条“跳过记录”：Status=skipped_mem，其它列置 missing
function _write_skip_row!(results_csv::String; fm, case_name, merging, alpha, perturbation, idno, load_id)
    vals = Dict{String,Any}(
        "Formulation"   => string(fm),
        "Perturbation"  => perturbation,
        "Case"          => case_name,
        "Merge"         => merging,
        "A_parameter"   => alpha,
        "Status"        => "skipped_mem",
        "ID"            => (ismissing(idno) ? -1 : idno),
        "load_id"       => load_id,
    )
    header = isfile(results_csv) ? open(io -> split(chomp(readline(io)), ','), results_csv) : EXPECTED_ORDER
    df = DataFrame()
    for col in header
        df[!, Symbol(col)] = [ get(vals, col, missing) ]
    end
    if isfile(results_csv)
        CSV.write(results_csv, df; append=true)
    else
        CSV.write(results_csv, df)
    end
end


using PowerModels
using InfrastructureModels
using JSON
using Printf

# --- parse <case>_<k>_perturbation_<seed>_<id>.json → (k, seed, id) ---
function _parse_k_seed_id(fname::String)
    m = match(r"^[A-Za-z0-9]+_([0-9]+(?:\.[0-9]+)?)_perturbation_([0-9]+)_([0-9]+)\.json$", fname)
    if m === nothing
        return (NaN, missing, missing)
    end
    k     = try parse(Float64, m.captures[1]) catch; NaN; end
    seed  = try parse(Int,     m.captures[2]) catch; missing; end
    idno  = try parse(Int,     m.captures[3]) catch; missing; end
    return (k, seed, idno)
end

_fmt_k(k::Float64) = isnan(k) ? nothing : @sprintf("%.2f", k)

# --- conservative memory estimator based on clique sizes ---
function _estimate_mem_gb(data, model, clique_merging::Bool, alpha::Float64)
    pm = InfrastructureModels.InitializeInfrastructureModel(model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym)
    PowerModels.ref_add_core!(pm.ref)
    nw = collect(InfrastructureModels.nw_ids(pm, PowerModels.pm_it_sym))[1]
    _adj, cadj, _lookup, _sigma, _q = PowerModels._chordal_extension(pm, nw, clique_merging, alpha)
    cliques = PowerModels._maximal_cliques(cadj)
    # 主耗 ~ Σ s_i^2（块对角密集块），乘以常数系数以覆盖工作区
    kappa = 12.0
    bytes = 8.0 * sum(length(c)^2 for c in cliques) * kappa
    return bytes / 1e9
end

function run_one_case(json_path::String, fm::String, merging::Bool, alpha::Float64)
    # 固定案例（按你现有脚本）
    case_file  = "case1888rte.m"
    case_name  = replace(case_file, ".m" => "")

    # 载入基础网络 + 应用扰动
    data_path = "/home/goatoine/Documents/Lanyue/data/raw_data/$case_file"
    data      = PowerModels.parse_file(data_path)
    loads     = JSON.parsefile(json_path)
    for (idd, load) in loads["load"]
        data["load"][idd]["pd"] = load["pd"]
        data["load"][idd]["qd"] = load["qd"]
    end
    for (_gen_id, gen) in data["gen"]
        println("gen cost before: ", gen["cost"])
        gen["cost"] .= gen["cost"] ./ 1e3
        println("gen cost after: ", gen["cost"])
    end
    # 输出文件命名 token（保持与你之前一致）
    fname = basename(json_path)
    k, seed, idno = _parse_k_seed_id(fname)
    k_tok = _fmt_k(k)
    tokens = ["pglib_opf", case_name]
    if k_tok !== nothing && !ismissing(idno)
        append!(tokens, ["k", k_tok, string(idno), "perturbation.csv"])
    else
        append!(tokens, ["perturbation.csv"])
    end

    # —— 自动检测内存 & 预检预算 ——
    mem_total_gb = Sys.total_memory() / 1024^3
    # 预算取 80% RAM，但至少 4GB，且保留 2GB 给系统
    budget_gb = max(4.0, min(0.80 * mem_total_gb, mem_total_gb - 2.0))
    est_gb = _estimate_mem_gb(data, eval(Symbol(fm)), merging, alpha)
    println("Estimated mem for ", (fname, fm, merging, alpha), ": ",
            round(est_gb, digits=2), " GB  [RAM=", round(mem_total_gb, digits=2), " GB, budget=", round(budget_gb, digits=2), " GB]")
    if est_gb > budget_gb * EST_TOL
        @warn "[SKIP] predicted mem exceeds budget*tolerance" predicted=round(est_gb, digits=2) budget=round(budget_gb, digits=2) tol=EST_TOL json=fname fm=fm merging=merging alpha=alpha

        # —— 生成与 solver_wrappers 相同的 CSV 路径 ——
        # 若你的 solver_wrappers 写到 data/solve_time/…，把 "clique_stats" 改成 "solve_time"
        results_dir = joinpath("data", "clique_stats", case_name)
        mkpath(results_dir)
        results_csv = joinpath(results_dir, join(tokens, "_"))

        # 写一行“跳过”占位
        perturbation_tuple = (isnan(k) ? NaN : k, seed)
        _write_skip_row!(results_csv;
            fm = fm, case_name = case_name, merging = merging, alpha = alpha,
            perturbation = perturbation_tuple, idno = idno, load_id = fname)

        return
    end


    perturbation = (isnan(k) ? NaN : k, seed)

    # 调用求解；结果 CSV 在 SolverWrappers.solve 内部按原格式写入
    SolverWrappers.solve(
        data,
        eval(Symbol(fm)),
        merging,
        fname;
        alpha = alpha,
        id_name = fname,
        tokens = tokens,
        perturbation = perturbation,
        id_detect = (ismissing(idno) ? -1 : idno),
    )
end

# ---- CLI entry ----
json_file, fm, merging_str, alpha_str = ARGS
run_one_case(json_file, fm, merging_str == "true", parse(Float64, alpha_str))
