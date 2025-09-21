import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("../src_jl/solver_wrappers.jl")
using CSV, DataFrames
using PowerModels
using InfrastructureModels
using JSON
using Printf


function _parse_k_id(fname::String)
    m = match(r"^pglib_opf_[A-Za-z0-9]+_k_([0-9]+(?:\.[0-9]+)?)_([0-9]+)_perturbation\.json$", fname)
    if m === nothing
        return (NaN, missing)
    end
    k     = try parse(Float64, m.captures[1]) catch; NaN; end
    idno  = try parse(Int,     m.captures[2]) catch; missing; end
    return (k, idno)  
end

_fmt_k(k::Float64) = isnan(k) ? nothing : @sprintf("%.2f", k)

function run_one_case(case_name::String, json_path::String, fm::String, merging::Bool, alpha::Float64)
    # 固定案例（按你现有脚本）
    case_file  = "$(case_name).m"
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
    k,  idno = _parse_k_id(fname)
    k_tok = _fmt_k(k)
    tokens = ["pglib_opf", case_name]
    if k_tok !== nothing && !ismissing(idno)
        append!(tokens, ["k", k_tok, string(idno), "perturbation.csv"])
    else
        append!(tokens, ["perturbation.csv"])
    end


    perturbation = (isnan(k) ? NaN : k, idno)

    # 调用求解；结果 CSV 在 SolverWrappers.solve 内部按原格式写入
    SolverWrappers.solve(
        data,
        eval(Symbol(fm)),
        merging,
        case_name;
        alpha = alpha,
        id_name = fname,
        tokens = tokens,
        perturbation = perturbation,
        id_detect = (ismissing(idno) ? -1 : idno),
    )
end

# ---- CLI entry ----
function main()
    # 手动指定参数，方便 debug
    case_file = "case118"
    json_file = "/home/goatoine/Documents/Lanyue/data/load_profiles/case118/pglib_opf_case118_k_0.07_1_perturbation.json"
    fm        = "Chordal_MD"
    merging   = true
    alpha     = 3.5
    run_one_case(case_file, json_file, fm, merging, alpha)
end
#main()



# ---- CLI entry ----
case_name, json_file, fm, merging_str, alpha_str = ARGS
run_one_case(case_name, json_file, fm, merging_str == "true", parse(Float64, alpha_str))

