module SolverWrappers
using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using DataFrames
using CSV
using Ipopt
include("chordalvisual.jl")
import .ChordalVisualizer: visualize_fillin, edge_lists

export solve, solve_opf

function solve(data, model, clique_merging, case_name; alpha = 3)
    #case_name = "$(case_name)_$(formulation)_$(clique_merging)",请帮我把case_name和其他东西分开
    original_case_name = case_name
    # 从原始名称中拆分
    case_name = split(original_case_name, "_")[1]  # 只取第一部分，如 "case14"
    other_info = join(split(original_case_name, "_")[2:end], "_")  # 剩下的部分
    pm = InfrastructureModels.InitializeInfrastructureModel(model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym)
    PowerModels.ref_add_core!(pm.ref)
    nw = collect(InfrastructureModels.nw_ids(pm, pm_it_sym))[1]
    adj, cadj, lookup_index, sigma, q = PowerModels._chordal_extension(pm, nw, clique_merging, alpha)
    save_path = joinpath("result", "figure", "graph", "$(case_name)", "$(other_info)_fillin.png")

    visualize_fillin(adj, cadj; q=q, savepath=save_path)
    cliques = PowerModels._maximal_cliques(cadj)
    lookup_bus_index = Dict((reverse(p) for p = pairs(lookup_index)))
    groups = [[lookup_bus_index[gi] for gi in g] for g in cliques]
    pm.ext[:SDconstraintDecomposition] = PowerModels._SDconstraintDecomposition(groups, lookup_index, sigma)
    PowerModels.build_opf(pm)
    result = optimize_model!(pm, optimizer=Mosek.Optimizer)
    return result
end

function solve_opf(data, model, solver)
    return PowerModels.solve_opf(data, model, solver)
end

function save_result(relax, result, case_name, clique_merging, formulation, result_path)
    df = DataFrame(Case = [case_name],
                   Perturbation = [(0,0)],
                   relaxation = [relax],
                   Merge = [clique_merging],
                   Formulation = [formulation],
                   SolveTime = [result["solve_time"]],
                   Status = [result["termination_status"]],
                   objective = [result["objective"]],
                   SolutionStatus = [result["primal_status"]],
                   SolutionStatusD = [result["dual_status"]])
    mkpath(joinpath(result_path, case_name))
    file_path = joinpath(result_path, case_name, case_name * string(relax) * string(formulation) * string(clique_merging) * ".csv")
    CSV.write(file_path, df; append=true, header=false)
end
end # module