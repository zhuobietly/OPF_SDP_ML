using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using CSV
using DataFrames
using Random
using JSON

function solve(data, model, clique_merging)
    pm = InfrastructureModels.InitializeInfrastructureModel(model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym)
    PowerModels.ref_add_core!(pm.ref)
    #network
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
    # traversal all the nodes to build the SDP matrix
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
        return nothing
    end
    Random.seed!(perturb_loads[2])
    sum_active = sum([load["pd"] for (bus,load) in data["load"]])
    sum_reactive = sum([load["qd"] for (bus,load) in data["load"]])
    original_loads = deepcopy(data["load"])
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
    return original_loads
end

function unperturbate!(data, original_loads)
    """
    why need to unperturbate?
    """
    if original_loads === nothing
        return
    end
    for (bus,load) in data["load"]
        load["pd"] = original_loads[bus]["pd"]
        load["qd"] = original_loads[bus]["qd"]
    
    end
end



formulations = [Chordal_MFI, Chordal_AMD, Chordal_MD]
merging_options = [true, false]
σ = 0.5

case_name = splitext(case)[1]

header_df = DataFrame( Case = String[], Formulation = String[], Perturbation = Tuple{Float64, Int}[], Merge = Bool[], SolveTime = Float64[])
CSV.write(case_name*"_perturbation.csv", header_df)
data = PowerModels.parse_file(case)

for (_gen_id, gen) in data["gen"]
    # power-angle cost: cost = [c₂, c₁, c₀]
    println("gen cost before: ", gen["cost"])
    gen["cost"] .= gen["cost"] ./ 1e3  
    println("gen cost after: ", gen["cost"])
end

for i in 1:1
    exit_loop = false
    perturb_loads = (σ , i)
    # Perturb the loads
    original_loads = perturbate!(data, perturb_loads)
    for merging in merging_options
        for formulation in formulations
            try
                result = solve(data, formulation, merging)
                if result["termination_status"] == INFEASIBLE
                    println("Infeasible")
                    exit_loop = true
                    break
                end
                df = DataFrame(Formulation = ["SDP"], 
                            formulation = string(formulation),
                            Case = c, 
                            Merge = Merging, 
                            SolveTime = result["solve_time"], 
                            Status = result["termination_status"], 
                            objective = result["objective"], 
                            SolutionStatus = result["primal_status"])
                CSV.write("results"*c*formulation".csv", df, append=true, header=false)
            catch e
                println("Error encountered for $formulation, $case: $e. Skipping this combination.")
                exit_loop = true
                break
            end
        end
        if exit_loop
            break
        end
    end
    unperturbate!(data, original_loads)
end