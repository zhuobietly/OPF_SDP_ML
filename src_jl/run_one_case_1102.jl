#!/usr/bin/env julia
# import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
include(joinpath(@__DIR__, "solver_wrappers_1102.jl"))

using JSON, Dates, Printf
using PowerModels, InfrastructureModels

function _parse_k_id(fname::String)
    m = match(r"^pglib_opf_[A-Za-z0-9]+_k_([0-9]+(?:\.[0-9]+)?)_([0-9]+)_perturbation\.json$", fname)
    m === nothing && return (NaN, missing)
    k   = try parse(Float64, m.captures[1]) catch; NaN; end
    idn = try parse(Int,     m.captures[2]) catch; missing; end
    return (k, idn)
end

_fmt_k(k::Float64) = isnan(k) ? nothing : @sprintf("%.2f", k)

# sidecar: <base>__st__FM__true|false__ALPHA.json
function _sidecar(json_path::AbstractString, fm::AbstractString, merging::Bool, alpha::Float64)
    base = splitext(basename(json_path))[1]
    mstr = merging ? "true" : "false"
    return joinpath(dirname(json_path), @sprintf("%s__st__%s__%s__%.1f.json", base, fm, mstr, alpha))
end

function run_one_case(case_name::String, json_path::String, fm::String, merging::Bool, alpha::Float64)
    data_path = "/home/goatoine/Documents/Lanyue/data/raw_data/$(case_name).m"
    data      = PowerModels.parse_file(data_path)

    loads = JSON.parsefile(json_path)
    for (idd, load) in loads["load"]
        data["load"][idd]["pd"] = load["pd"]
        data["load"][idd]["qd"] = load["qd"]
    end
    for (_id, gen) in data["gen"]; gen["cost"] .= gen["cost"] ./ 1e3; end
    fname      = basename(json_path)
    k, idno    = _parse_k_id(fname)
    k_tok      = _fmt_k(k)
    tokens     = ["pglib_opf", case_name]
    if k_tok !== nothing && !ismissing(idno)
        append!(tokens, ["k", k_tok, string(idno), "perturbation.csv"])
    else
        append!(tokens, ["perturbation.csv"])
    end
    perturbation = (isnan(k) ? NaN : k, idno)
    file_name    = splitext(fname)[1]

    # 调用你的 solver（注意：你当前 solver 返回顺序为：df, solve_time, csv_path）
    status = "OK"; err = ""
    df = nothing; solve_time = NaN; csv_path = ""
    # try
    df, solve_time, csv_path = SolverWrappers.solve(
        data, eval(Symbol(fm)), merging, case_name;
        alpha=alpha, id_name=fname, tokens=tokens,
        perturbation=perturbation, id_detect=(ismissing(idno) ? -1 : idno),
        file_name=file_name
    )
    # catch e
    #     status = "FAIL"
    #     err    = sprint(showerror, e)
    # end

    # 写 sidecar：包含 solve_time + csv_path + 行选择键（便于后续删除该 load 的 15 行）
    sidecar = _sidecar(json_path, fm, merging, alpha)
    open(sidecar, "w") do io
        JSON.print(io, Dict(
            "timestamp"   => string(Dates.now()),
            "status"      => status,
            "error"       => err,
            "solve_time"  => solve_time,         # ✅ 用于选最快
            "csv_path"    => csv_path,           # ✅ 用于保留/删除
            "row_key"     => Dict(               # ✅ 用于定位 CSV 中的行
                "load_id"     => fname,
                "Formulation" => fm,
                "Merge"       => merging,
                "A_parameter" => alpha,
            ),
            "case"        => case_name,
            "json"        => json_path,
            "fm"          => fm,
            "merging"     => merging,
            "alpha"       => alpha,
            "k"           => (isnan(k) ? nothing : k),
            "id"          => (ismissing(idno) ? nothing : idno),
        ), 2)
    end
end

# if abspath(PROGRAM_FILE) == @__FILE__
#     case_name, json_file, fm, merging_str, alpha_str = ARGS
#     run_one_case(case_name, json_file, fm, merging_str == "true", parse(Float64, alpha_str))
# end
function __main__(args)
    case_name, json_file, fm, merging_str, alpha_str = args
    run_one_case(case_name, json_file, fm, merging_str == "true", parse(Float64, alpha_str))
end

# 不再比较 PROGRAM_FILE，直接看是否有参数
if !isempty(ARGS)
    __main__(ARGS)
end
