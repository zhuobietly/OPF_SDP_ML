# -----------------------------
# main_experiment.jl（主入口脚本）
include("../src_jl/perturbation.jl")
include("../src_jl/solver_wrappers.jl")

using .Perturbation
using .SolverWrappers
using PowerModels
using CSV, DataFrames
using Ipopt
@show isdefined(SolverWrappers, :solve_opf)

formulations = [Chordal_MFI, Chordal_AMD, Chordal_MD]
merging_options = [true, false]
scdir = @__DIR__
case_dir = joinpath(dirname(scdir), "data", "raw_data")
cases = ["case14.m"]
result_path = joinpath(dirname(scdir), "result", "csv", "gap_experim_new")
solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)

for case in cases
    case_path = joinpath(case_dir, case)
    case_name = splitext(case)[1]
    data = PowerModels.parse_file(case_path)
    perturbate!(data, (0, 0))

    for (_gen_id, gen) in data["gen"]
        println("gen cost before: ", gen["cost"])
        gen["cost"] .= gen["cost"] ./ 1e3
        println("gen cost after: ", gen["cost"])
    end

    # Run AC, SOC, QC
    # for (relax_label, model) in zip(
    #     ["AC", "SOC", "QC"],
    #     [ACPPowerModel, SOCWRPowerModel, QCRMPowerModel]
    # )
    #     result = SolverWrappers.solve_opf(data, model, solver)
    #     SolverWrappers.save_result(relax_label, result, case_name, "None", "None", result_path)
    # end

    # Run SDP variants
    for formulation in formulations
        for clique_merging in merging_options
            try
                save_name = "$(case_name)_$(formulation)_$(clique_merging)"
                result = SolverWrappers.solve(data, formulation, clique_merging, save_name)
                #function save_result(relax, result, case_name, clique_merging, formulation, result_path)
                SolverWrappers.save_result("SDP", result, case_name, clique_merging, formulation, result_path)
            catch e
                println("Error $e encountered for $formulation, $case. Skipping this combination.")
            end
        end
    end
end
