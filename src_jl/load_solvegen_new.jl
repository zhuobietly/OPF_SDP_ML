using CSV, DataFrames, Printf

# 与现有 CSV 一致的列顺序（按你现在的表头）
const EXPECTED_ORDER = [
    "network_type","Formulation","Perturbation","Case","Merge","A_parameter","SolveTime",
    "Status","objective","SolutionStatus","ID","load_id",
    "Iterations","PrimalRes","DualRes","RelGap","KKTCondProxy","ActiveLimits",
    "r_max","t","sum_r_sq","sum_r_cu","sep_max","sep_mean","sum_sep_sq",
    "tree_max_deg","tree_h","fillin","coupling"
]

# 解析 <case>_<k>_perturbation_<seed>_<id>.json
function _parse_k_seed_id(fname::String)
    m = match(r"^[A-Za-z0-9]+_([0-9]+(?:\\.[0-9]+)?)_perturbation_([0-9]+)_([0-9]+)\\.json$", fname)
    if m === nothing
        return (NaN, missing, missing)
    end
    k     = try parse(Float64, m.captures[1]) catch; NaN; end
    seed  = try parse(Int,     m.captures[2]) catch; missing; end
    idno  = try parse(Int,     m.captures[3]) catch; missing; end
    return (k, seed, idno)
end

# 往目标 CSV 里追加一行状态记录（status 可为 "killed_mem" / "skipped_mem"）
function _write_status_row!(results_csv::String; fm, case_name, merging, alpha, perturbation, idno, load_id, status::String)
    vals = Dict{String,Any}(
        "Formulation"   => string(fm),
        "Perturbation"  => perturbation,
        "Case"          => case_name,
        "Merge"         => merging,
        "A_parameter"   => alpha,
        "Status"        => status,
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


case_name  = "case118"  # "case300_ieee"  # "case1354pegase"  # "case2383wp"  # "case6468rte"  # "case9241pegase"
input_dir  = joinpath("/home/goatoine/Documents/Lanyue/data/load_profiles/", case_name)

# 策略组合
formulations = ["Chordal_MD", "Chordal_MFI", "Chordal_AMD"]
merging_opts = ["false", "true"]
alpha_values = [2.0,2.5,3.0,3.5,4,4.5,5.0]
# —— 自动检测机器内存，给每个子进程加软上限 ——
mem_total_gb = try Sys.total_memory() / 1024^3 catch; 16.0 end

# —— 放宽并支持 ENV 覆盖（只改这三行） ——
mem_soft_frac   = try parse(Float64, get(ENV, "MEM_SOFT_FRAC", "0.97")) catch; 0.998 end   # 默认 97% RAM
mem_soft_res_gb = try parse(Float64, get(ENV, "MEM_SOFT_RES_GB", "0.5"))  catch; 0.15  end  # 默认至少留 0.5 GB 给系统
mem_soft_gb     = max(6.0, min(mem_soft_frac * mem_total_gb, mem_total_gb - mem_soft_res_gb))

mem_kb = 19689532 # ulimit 以 KB 为单位
# mem_kb = Int(floor(mem_soft_gb * 1024^2))  # 转成 KB
println("RAM=", round(mem_total_gb,digits=2)," GB, cap=", round(mem_soft_gb,digits=2)," GB, ulimit -Sv=", mem_kb, " KB ")

for json_file in readdir(input_dir)
    endswith(json_file, ".json") || continue
    filepath = joinpath(input_dir, json_file)
    for fm in formulations
        for merging in merging_opts
            alphas = (merging == "true") ? alpha_values : [0.0]
            for alpha in alphas
                # 用 bash 设置 ulimit -Sv（虚拟内存软上限，单位 KB），然后 exec 进入 julia 子进程
                shcmd = "ulimit -Sv $(mem_kb); exec julia --project=.. src_jl/run_one_case.jl '$case_name' '$filepath' $(fm) $(merging) $(alpha)"
                try
                    run(`bash -lc $shcmd`)
                catch e
                    @warn "Subprocess failed (OOM/ulimit). Recording and skipping." file=filepath fm=fm merging=merging alpha=alpha err=e

                    # —— 构造与 run_one_case/solver_wrappers 相同的 CSV 路径 ——
                    fname = basename(filepath)
                    k, seed, idno = _parse_k_seed_id(fname)

                    # 如果你的 solver_wrappers 写到 data/solve_time/…，把 "clique_stats" 改成 "solve_time"
                    k_tok = isnan(k) ? nothing : @sprintf("%.2f", k)
                    tokens = ["pglib_opf", case_name]
                    if k_tok !== nothing && !ismissing(idno)
                        append!(tokens, ["k", k_tok, string(idno), "perturbation.csv"])
                    else
                        append!(tokens, ["perturbation.csv"])
                    end
                    results_dir = joinpath("data", "clique_stats", case_name)   # ← 若用 solve_time 就改这里
                    mkpath(results_dir)
                    results_csv = joinpath(results_dir, join(tokens, "_"))

                    # 写一行状态：运行期被内存限制/系统杀掉
                    perturbation_tuple = (isnan(k) ? NaN : k, seed)
                    _write_status_row!(results_csv;
                        fm = fm,
                        case_name = case_name,
                        merging = (merging == "true"),
                        alpha = alpha,
                        perturbation = perturbation_tuple,
                        idno = idno,
                        load_id = fname,
                        status = "killed_mem",
                    )
                end

            end
        end
    end
end

println("✅ All cases completed (auto RAM detection + per-run soft cap).")
