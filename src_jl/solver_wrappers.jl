module SolverWrappers
using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using DataFrames
using CSV
using Ipopt
using JuMP
include("../src_jl/chordalvisual.jl")
using .ChordalVisualizer: visualize_fillin, edge_lists
include("../src_jl/ChordalStatsLite.jl")
using .ChordalStatsLite: compute_stats_from_vars,
                        cliques_from_peo,
                        stats_dataframe,
                        append_stats_csv
include("../src_jl/LightGC.jl")
using .LightGC: cleanup!, safe_close

export solve, solve_opf
# 用 Pipe 捕获 stdout/stderr（跨 Julia 版本稳）
# ---------------- 捕获 stdout/stderr：管道优先，缺失则用临时文件 ----------------
const _HAS_PIPE = isdefined(Base, :pipe)

# --- 捕获 stdout/stderr 到字符串（用临时文件，跨版本稳） ---
function _run_with_capture(f::Function)
    outpath = tempname(); errpath = tempname()
    ret = open(outpath, "w") do outio
        open(errpath, "w") do errio
            redirect_stdout(outio) do
                redirect_stderr(errio) do
                    f()
                end
            end
        end
    end
    out_s = isfile(outpath) ? read(outpath, String) : ""
    err_s = isfile(errpath) ? read(errpath, String) : ""
    try rm(outpath; force=true); rm(errpath; force=true) catch; end
    return ret, string(out_s, "\n", err_s)
end

# --- 解析 Mosek 日志：Iterations / Primal / Dual / Relative gap ---
# 解析 Mosek 日志：优先读显式 Summary；否则读 ITE 表格最后一行
# 返回 (iters::Union{Int,Missing}, pfeas::Union{Float64,Missing},
#        dfeas::Union{Float64,Missing}, relgap::Union{Float64,Missing}, time_sec::Union{Float64,Missing})
function parse_mosek_log_all(logtxt::AbstractString)
    # ---- 1) 显式 Summary 行（Primal/Dual infeasibility, Relative gap, Iterations, Time）----
    iters = try
        m = match(r"Iterations:\s+(\d+)", logtxt); m === nothing ? missing : parse(Int, m.captures[1])
    catch; missing; end
    pfeas = try
        m = match(r"Primal\s+(?:in)?feasibility\s*[:=]\s*([0-9.eE+\-]+)", logtxt)
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch; missing; end
    dfeas = try
        m = match(r"Dual\s+(?:in)?feasibility\s*[:=]\s*([0-9.eE+\-]+)", logtxt)
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch; missing; end
    relgap = try
        m = match(r"Relative\s+gap\s*[:=]\s*([0-9.eE+\-]+)", logtxt)
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch; missing; end
    time_sec = try
        m = match(r"Optimizer terminated\. Time:\s*([0-9.]+)", logtxt)
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch; missing; end

    # ---- 2) 若没找到 Summary，就从 ITE 表格最后一行取值 ----
    if any(ismissing, (iters, pfeas, dfeas, relgap))
        last = nothing
        for ln in eachline(IOBuffer(logtxt))
            # 形如： "16  9.0e-09  3.9e-09  2.8e-13  1.00e+00   6.097266787e+00   6.097266781e+00   9.7e-10  0.01"
            m = match(r"^\s*(\d+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+[0-9.eE+\-]+\s+\S+\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+[0-9.eE+\-]+\s+([0-9.]+)\s*$", ln)
            if m !== nothing
                last = m
            end
        end
        if last !== nothing
            iters = ismissing(iters) ? parse(Int, last.captures[1]) : iters
            pfeas = ismissing(pfeas) ? parse(Float64, last.captures[2]) : pfeas
            dfeas = ismissing(dfeas) ? parse(Float64, last.captures[3]) : dfeas
            # 相对间隙=|POBJ-DOBJ|/max(1,|POBJ|)
            if ismissing(relgap)
                pobj = parse(Float64, last.captures[4])
                dobj = parse(Float64, last.captures[5])
                relgap = abs(pobj - dobj) / max(1.0, abs(pobj))
            end
            time_sec = ismissing(time_sec) ? parse(Float64, last.captures[6]) : time_sec
        end
    end
    return iters, pfeas, dfeas, relgap, time_sec
end

# ----------------------- 小工具：健壮取数 -----------------------
# 兼容 String/Symbol 键
_get_any(dict, k, default=missing) = haskey(dict, k) ? dict[k] : default
_get_any(dict, ks::Vector, default=missing) = begin
    for k in ks
        if haskey(dict, k); return dict[k]; end
    end
    return default
end

# 从字典读取数值（Number），否则返回 missing
function _getnum(d, keysets...; default=missing)
    for ks in keysets
        v = _get_any(d, ks, nothing)
        if v isa Number; return Float64(v); end
    end
    return default
end

# 嵌套读取（如 result["solution"]）
function _get_nested(d, path::Vector{Any}, default=missing)
    cur = d
    for k in path
        if !(cur isa AbstractDict) || !haskey(cur, k); return default; end
        cur = cur[k]
    end
    return cur
end

# KKT 条件的“尺度代理”：用网络数据里可拿到的系数/物理量构造 max/min 比
function _coeff_ratio_from_data(data; tol=1e-16)
    coeffs = Float64[]
    # buses
    bus = get(data, "bus", get(data, :bus, Dict()))
    for (_k, b) in bus
        for key in ("pd","qd","gs","bs","gsh","bsh")
            v = get(b, key, get(b, Symbol(key), nothing))
            v isa Number && push!(coeffs, abs(float(v)))
        end
    end
    # branches
    brs = get(data, "branch", get(data, :branch, Dict()))
    for (_k, br) in brs
        for key in ("r","x","b","g","br_b","br_r","br_x")
            v = get(br, key, get(br, Symbol(key), nothing))
            v isa Number && push!(coeffs, abs(float(v)))
        end
        # 线路额定也反映量级
        for key in ("rate_a","rate_b","rate_c")
            v = get(br, key, get(br, Symbol(key), nothing))
            v isa Number && push!(coeffs, abs(float(v)))
        end
    end
    # 生成机上下界（量纲）
    gens = get(data, "gen", get(data, :gen, Dict()))
    for (_k, g) in gens
        for key in ("pmax","pmin","qmax","qmin")
            v = get(g, key, get(g, Symbol(key), nothing))
            v isa Number && push!(coeffs, abs(float(v)))
        end
    end
    coeffs = filter(!iszero, coeffs)
    return isempty(coeffs) ? missing :
           (maximum(coeffs) / max(minimum(coeffs), tol))
end

# 贴边计数（电压、发电机 P/Q、线路流）——尽力而为，拿不到就 missing
function _count_active_limits(result, data; tol=1e-4)
    sol = _get_any(result, ["solution", :solution], nothing)
    sol isa AbstractDict || return missing
    cnt = 0

    # bus voltage
    sbus = _get_any(sol, ["bus", :bus], Dict())
    dbus = get(data, "bus", get(data, :bus, Dict()))
    for (id, bv) in sbus
        vm = _get_any(bv, ["vm", :vm], nothing)
        vm isa Number || continue
        bd = get(dbus, string(id), get(dbus, id, nothing))
        bd isa AbstractDict || continue
        vmin = _get_any(bd, ["vmin", :vmin], nothing)
        vmax = _get_any(bd, ["vmax", :vmax], nothing)
        (vmin isa Number && abs(vm - vmin) <= tol) && (cnt += 1)
        (vmax isa Number && abs(vm - vmax) <= tol) && (cnt += 1)
    end

    # generators P/Q
    sgen = _get_any(sol, ["gen", :gen], Dict())
    dgen = get(data, "gen", get(data, :gen, Dict()))
    for (id, gv) in sgen
        gd = get(dgen, string(id), get(dgen, id, nothing))
        gd isa AbstractDict || continue
        pg = _get_any(gv, ["pg", :pg], nothing)
        qg = _get_any(gv, ["qg", :qg], nothing)
        if pg isa Number
            pmin = _get_any(gd, ["pmin", :pmin], nothing)
            pmax = _get_any(gd, ["pmax", :pmax], nothing)
            (pmin isa Number && abs(pg - pmin) <= tol) && (cnt += 1)
            (pmax isa Number && abs(pg - pmax) <= tol) && (cnt += 1)
        end
        if qg isa Number
            qmin = _get_any(gd, ["qmin", :qmin], nothing)
            qmax = _get_any(gd, ["qmax", :qmax], nothing)
            (qmin isa Number && abs(qg - qmin) <= tol) && (cnt += 1)
            (qmax isa Number && abs(qg - qmax) <= tol) && (cnt += 1)
        end
    end

    # branch flow (active power) vs rate_a
    sbr = _get_any(sol, ["branch", :branch], Dict())
    dbr = get(data, "branch", get(data, :branch, Dict()))
    for (id, bv) in sbr
        pf = _get_any(bv, ["pf", :pf], nothing)
        bd = get(dbr, string(id), get(dbr, id, nothing))
        ratea = (bd isa AbstractDict) ? _get_any(bd, ["rate_a", :rate_a], nothing) : nothing
        if pf isa Number && ratea isa Number
            abs(abs(pf) - ratea) <= tol && (cnt += 1)
        end
    end

    return cnt
end

function solve(data, model, clique_merging, case_name; alpha = 3, id_name = nothing, tokens = nothing, perturbation = nothing, id_detect = -1)
    #case_name = "$(case_name)_$(formulation)_$(clique_merging)",请帮我把case_name和其他东西分开
    original_case_name = case_name
    # 从原始名称中拆分
    case_name = split(original_case_name, "_")[1]  # 只取第一部分，如 "case14"
    other_info = join(split(original_case_name, "_")[2:end], "_")  # 剩下的部分
    pm = InfrastructureModels.InitializeInfrastructureModel(model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym)
    PowerModels.ref_add_core!(pm.ref)
    nw = collect(InfrastructureModels.nw_ids(pm, pm_it_sym))[1]
    adj, cadj, lookup_index, sigma, q = PowerModels._chordal_extension(pm, nw, clique_merging, alpha)
    @assert q == invperm(sigma) "置换不一致：按理应当满足 q == invperm(sigma)"
    #原始矩阵画图
    save_path = joinpath("result", "figure", "graph", "$(case_name)", "$(other_info)_fillin.png")
    visualize_fillin(adj, cadj; q=q, savepath=save_path)
    println("✅ 绘制完成原始顺序）：", save_path)
    # Step 3: PEO 顺序的图
    save_path_peo = joinpath("result", "figure", "graph", case_name, "$(other_info)_fillin_peo.png")
    ChordalVisualizer.visualize_fillin(adj, cadj; q=sigma, savepath=save_path_peo)
    println("✅ 绘制完成（PEO 顺序）：", save_path_peo)
    cliques = PowerModels._maximal_cliques(cadj)
    lookup_bus_index = Dict((reverse(p) for p = pairs(lookup_index)))
    groups = [[lookup_bus_index[gi] for gi in g] for g in cliques]
    pm.ext[:SDconstraintDecomposition] = PowerModels._SDconstraintDecomposition(groups, lookup_index, sigma)
    # ========= 结构统计（求解前已完成）=========
    stats = compute_stats_from_vars(; cadj=cadj, sigma=sigma, cliques=cliques, cadj0=adj)

    PowerModels.build_opf(pm)
    #result = optimize_model!(pm, optimizer=Mosek.Optimizer)
    # ===== 求解 =====
    # 用 optimizer_with_attributes 确保 Mosek 输出日志（不要依赖 set_optimizer_attribute）
    opt = optimizer_with_attributes(
        Mosek.Optimizer,
        "MSK_IPAR_LOG" => 1,
        "MSK_IPAR_LOG_INTPNT" => 1,
        "QUIET" => 0,          # JuMP 层静音开关
    )

    # 运行并捕获日志
    result, mosek_log = _run_with_capture() do
        optimize_model!(pm, optimizer=opt)
    end

    # -- MOI 读取迭代步（优先用 MOI；拿不到再用日志） （// NEW）
    log_iters, log_pr, log_dr, log_rg, log_time = parse_mosek_log_all(mosek_log)

    iterations = log_iters
    primal_res = log_pr
    dual_res   = log_dr
    rel_gap    = log_rg
    mosektime  = log_time

    # （可选）把日志写文件，方便你确认到底捕到了什么
    
    try
        # 你的主逻辑
        mkpath("runs/chordalstats_logs")
        open(joinpath("runs","chordalstats_logs","$(case_name)_$(other_info).log"), "w") do io
            write(io, mosek_log)
        end
    finally
        cleanup!(Ref(pm), Ref(adj), Ref(cadj), Ref(mosek_log))
    end
    
    # —— 兼容 Symbol / String 键 —— 
    _get(r, ks, default=missing) = begin
        for k in ks
            if haskey(r, k); return r[k]; end
        end
        return default
    end

    solve_time  = _get(result, ["solve_time", :solve_time], NaN)
    term_status = string(_get(result, ["termination_status", :termination_status, "status", :status], ""))
    obj_val     = _get(result, ["objective", :objective, "obj_val", :obj_val], NaN)
    sol_status  = string(_get(result, ["solution_status", :solution_status, "primal_status", :primal_status], ""))
    ##
    # 若没直接给相对间隙，尝试用上下界/对偶目标估计
    if rel_gap === missing
        obj_lb = _getnum(result, ["objective_lb", :objective_lb, "best_bound", :best_bound,
                                  "dual_objective", :dual_objective])
        if !(obj_val === missing || obj_lb === missing)
            denom = max(1.0, abs(obj_val))
            rel_gap = abs(obj_val - obj_lb) / denom
        end
    end
    # KKT 条件 proxy（基于数据的量纲范围）
    kkt_cond_proxy = _coeff_ratio_from_data(data)

    # 贴边数量统计（有解且数据齐全才会返回 Int，否则 missing）
    active_limits = _count_active_limits(result, data; tol=1e-4)


    # —— 你要的“求解结果”列（放前面）——
    df_core = DataFrame(
        Formulation     = [string(model)],
        Perturbation    = [perturbation],     # 例：(σ, seed)
        Case            = [case_name],
        Merge           = [clique_merging],
        A_parameter     = [alpha],
        SolveTime       = [solve_time],
        mosektime       = [mosektime],
        Status          = [term_status],
        objective       = [obj_val],
        SolutionStatus  = [sol_status],
        ID              = [id_detect],
        load_id         = [id_name],
        # 新增数值/收敛列
        Iterations      = [iterations],
        PrimalRes       = [primal_res],
        DualRes         = [dual_res],
        RelGap          = [rel_gap],
        KKTCondProxy    = [kkt_cond_proxy],
        ActiveLimits    = [active_limits],
    )

    # —— 结构统计（紧跟在后面）——
    df_stats = DataFrame(
        r_max        = [stats.r_max],
        t            = [stats.t],
        sum_r_sq     = [stats.sum_r_sq],
        sum_r_cu     = [stats.sum_r_cu],
        sep_max      = [stats.sep_max],
        sep_mean     = [stats.sep_mean],
        sum_sep_sq   = [stats.sum_sep_sq],
        tree_max_deg = [stats.tree_max_degree],
        tree_h       = [stats.tree_height],
        fillin       = [stats.fillin_ratio],
        coupling     = [stats.coupling_proxy],
    )

    # —— 合并成一行 —— 
    df = hcat(df_core, df_stats)

    # —— 生成 CSV 路径（修正逻辑）——
    stats_csv_path = joinpath("data", "clique_stats", case_name, join(tokens, "_"))
    mkpath(dirname(stats_csv_path))

    # —— 追加写入 —— 
    if isfile(stats_csv_path)
        CSV.write(stats_csv_path, df; append=true)
    else
        CSV.write(stats_csv_path, df)
    end

    return result
end

function solve_opf(data, model, solver)
    return PowerModels.solve_opf(data, model, solver)
end


end # module