using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using CSV
using DataFrames
using Random
using Ipopt

function solve(data, model, clique_merging, perturb_loads=(0,0))
    pm = InfrastructureModels.InitializeInfrastructureModel(model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym)
    PowerModels.ref_add_core!(pm.ref)
    nw = collect(InfrastructureModels.nw_ids(pm, pm_it_sym))[1]
    println("Beginning chordal extension")
    cadj, lookup_index, sigma = PowerModels._chordal_extension(pm, nw, clique_merging)
    cliques = PowerModels._maximal_cliques(cadj)
    lookup_bus_index = Dict((reverse(p) for p = pairs(lookup_index)))
    groups = [[lookup_bus_index[gi] for gi in g] for g in cliques]
    # for (i, g) in enumerate(groups)
    #     println("Group ", i, ": ", g)
    # end
    # size = PowerModels._problem_size(groups) 

    pm.ext[:SDconstraintDecomposition] = PowerModels._SDconstraintDecomposition(groups, lookup_index, sigma)

    println("Building the opf")
    PowerModels.build_opf(pm)
    result = optimize_model!(pm, optimizer=Mosek.Optimizer)

    println("Size of the problem: ", size)
    println("Objective value: ", result["objective"])
    println("Status: ", result["termination_status"])
    println("Solve time: ", result["solve_time"], "s")
    return result
end

function perturbate!(data,perturb_loads=(0,0))
    if perturb_loads[1] == 0
        return
    end
    Random.seed!(perturb_loads[2])
    sum_active = sum([load["pd"] for (bus,load) in data["load"]])
    sum_reactive = sum([load["qd"] for (bus,load) in data["load"]])
    for (bus,load) in data["load"]
        load["pd"] += randn() * abs(perturb_loads[1] * load["pd"])
        load["pd"] = max(0, load["pd"])
        load["qd"] += randn() * abs(perturb_loads[1] * load["qd"])
        load["qd"] = max(0, load["qd"])
    end
    sum_active_new = sum([load["pd"] for (bus,load) in data["load"]])
    sum_reactive_new = sum([load["qd"] for (bus,load) in data["load"]])
    diff_active = sum_active_new - sum_active
    diff_reactive = sum_reactive_new - sum_reactive
    println("A total of ", diff_active, " MW and ", diff_reactive, " MVar were added to the system")
end


formulations = [Chordal_MFI, Chordal_AMD, Chordal_MD]
merging_options = [true, false]
#case directory
case_dir = joinpath("/home/goatoine/Documents/Lanyue/data/data/raw_data")
cases = [filename for filename in readdir(case_dir) if endswith(filename, ".m")]
cases = ["case14.m","case118.m","case1888rte.m"]
scdir = @__DIR__
merging_options = [true, false]
csv_path = joinpath(scdir, "/home/goatoine/Documents/Lanyue/result/gap_experim/")

#=header_SDR_df = DataFrame(Case = String[],
                        Perturbation = Tuple{Float64, Int}[],
                        Merge = Bool[],
                        Formulation = String[],
                        SolveTime = Float64[], 
                        Status = Any[], 
                        objective = Float64[], 
                        SolutionStatus = Any[],
                        SolutionStatusD = Any[])


CSV.write(csv_path, header_SDR_df)=#

# for case in cases
#     path = joinpath(scdir, "../data/Victor_data/", case)
#     if isfile(path) || isdir(path)
#         println("✅ Le chemin existe et est valide : $path")
#     else
#         println("❌ Le chemin n'existe pas ou n'est pas accessible : $path")
#     end
# end
solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)

for case in cases
    case_path = joinpath(scdir, "/home/goatoine/Documents/Lanyue/data/data/raw_data/", case)
    case_name = splitext(case)[1]
    data = PowerModels.parse_file(case_path)
    perturbate!(data, (0, 0))
    for (_gen_id, gen) in data["gen"]
        # power-angle cost: cost = [c₂, c₁, c₀]
        println("gen cost before: ", gen["cost"])
        gen["cost"] .= gen["cost"] ./ 1e3  
        println("gen cost after: ", gen["cost"])
    end
    result  = solve_opf(data,   ACPPowerModel, solver)
                df_SDR = DataFrame(Case = [case_name],
                            Perturbation = [(0,0)],
                            Merge = ["None"],
                            Formulation = ["AC"],
                            SolveTime = [result["solve_time"]], 
                            Status = [result["termination_status"]], 
                            objective = [result["objective"]], 
                            SolutionStatus = [result["primal_status"]],
                            SolutionStatusD =[result["dual_status"]])
                mkpath(joinpath(csv_path, case_name))
                CSV.write(joinpath(csv_path, case_name, case_name*"AC.csv"), df_SDR; header=false, append=true)
    result = solve_opf(data, SOCWRPowerModel, solver)
                df_SDR = DataFrame(Case = [case_name],
                            Perturbation = [(0,0)],
                            Merge = ["None"],
                            Formulation = ["SOC"],
                            SolveTime = [result["solve_time"]], 
                            Status = [result["termination_status"]], 
                            objective = [result["objective"]], 
                            SolutionStatus = [result["primal_status"]],
                            SolutionStatusD =[result["dual_status"]])
                mkpath(joinpath(csv_path, case_name))
                CSV.write(joinpath(csv_path, case_name, case_name*"SOC.csv"), df_SDR; header=false, append=true)
    result = solve_opf(data,  QCRMPowerModel, solver)
                df_SDR = DataFrame(Case = [case_name],
                            Perturbation = [(0,0)],
                            Merge = ["None"],
                            Formulation = ["QC"],
                            SolveTime = [result["solve_time"]], 
                            Status = [result["termination_status"]], 
                            objective = [result["objective"]], 
                            SolutionStatus = [result["primal_status"]],
                            SolutionStatusD =[result["dual_status"]])
                mkpath(joinpath(csv_path, case_name))
                CSV.write(joinpath(csv_path, case_name, case_name*"QC.csv"), df_SDR; header=false, append=true)
    for formulation in formulations
        for clique_merging in merging_options
            try
                result = solve(data, formulation, clique_merging)
                df_SDR = DataFrame(Case = [case_name],
                            Perturbation = [(0,0)],
                            Merge = [clique_merging],
                            Formulation = [formulation],
                            SolveTime = [result["solve_time"]], 
                            Status = [result["termination_status"]], 
                            objective = [result["objective"]], 
                            SolutionStatus = [result["primal_status"]],
                            SolutionStatusD =[result["dual_status"]])
                mkpath(joinpath(result_path, case_name))
                CSV.write(joinpath(result_path, case_name, case_name*"SDP"*string(formulation)*string(clique_merging)*".csv"), df_SDR, append=true, header=false)
            catch e
                println("Error $e encountered for $formulation, $case: . Skipping this combination.")
                continue
            end

        end
    end
end
#=
# Verrou pour les écritures concurrentes
const csv_lock = ReentrantLock()

# Boucle parallèle : chaque thread prend un indice dans `cases`
@threads for idx in 1:length(cases)
    case = cases[idx]
    case_path = joinpath(scdir, "../data/raw_data/", case)
    case_name = splitext(case)[1]
    data = PowerModels.parse_file(case_path)
    # perturbation nulle pour initialiser
    perturbate!(data, (0,0))

    # Stocker localement les lignes pour ce cas
    results = DataFrame(
        Case             = String[],
        Perturbation     = Tuple{Float64, Int}[],
        Merge            = Bool[],
        Formulation      = String[],
        SolveTime        = Float64[], 
        Status           = Any[], 
        objective        = Float64[], 
        SolutionStatus   = Any[],
        SolutionStatusD  = Any[],
    )

    for formulation in formulations
        for clique_merging in merging_options
            try
                result = solve(data, formulation, clique_merging)
                push!(results, (
                    Case            = case_name,
                    Perturbation    = (0,0),
                    Merge           = clique_merging,
                    Formulation     = formulation,
                    SolveTime       = result["solve_time"],
                    Status          = result["termination_status"],
                    objective       = result["objective"],
                    SolutionStatus  = result["primal_status"],
                    SolutionStatusD = result["dual_status"],
                ))
            catch e
                @warn "Erreur $e pour formulation=$formulation, case=$case_name. Ignoré."
            end
        end
    end

    # Écriture thread-safe
    lock(csv_lock) do
        CSV.write(csv_path, results; append=true, header=false)
    end
end
=#