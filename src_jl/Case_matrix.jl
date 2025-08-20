using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using CSV
using DataFrames
using Random
using JSON
using Dates
using JLD2

function save_sparse_matrix(cadj, filename)
    # Save the sparse matrix to a file in JLD2 format
    @save filename cadj
end

function print_memory_usage()
    println("🧠  Mem usage: ", round(Sys.total_memory() / 1024^3, digits=2), " GB total | ",
            round(Sys.free_memory() / 1024^3, digits=2), " GB free")
end

function solve(data, model, clique_merging)
    pm = InfrastructureModels.InitializeInfrastructureModel(model, data, PowerModels._pm_global_keys, PowerModels.pm_it_sym)
    PowerModels.ref_add_core!(pm.ref)

    println(pm)

    nw = collect(InfrastructureModels.nw_ids(pm, pm_it_sym))[1]
    println("Beginning chordal extension")
    cadj1, lookup_index1 = _adjacency_matrix(pm, nw)

    cadj, lookup_index, sigma = PowerModels._chordal_extension(pm, nw, clique_merge = clique_merging)
    cliques = PowerModels._maximal_cliques(cadj)
    lookup_bus_index = Dict((reverse(p) for p in pairs(lookup_index)))
    groups = [[lookup_bus_index[gi] for gi in g] for g in cliques]

    pm.ext[:SDconstraintDecomposition] = PowerModels._SDconstraintDecomposition(groups, lookup_index, sigma)

    println("Building the opf")
    PowerModels.build_opf(pm)
    result = optimize_model!(pm, optimizer=Mosek.Optimizer)

    println("Objective value: ", result["objective"])
    println("Status: ", result["termination_status"])
    println("Solve time: ", result["solve_time"], "s")
    return result, cadj1, lookup_index1
end

# --- Script principal ---

current_dir = @__DIR__
println("Current directory: ", current_dir)

# Paramètres du cas
case_file  = "case118.m"       # Nom du fichier de cas
num_suffix = "_test"            # Suffixe pour distinguer ce run

# Dossiers de données et de sorties
input_dir  = joinpath(current_dir, "output", "PL_118_10")
matrix_dir = joinpath(current_dir,
    "Results_" * replace(case_file, "." => "_") * num_suffix)

# Création du dossier de matrices s'il n'existe pas
if !isdir(matrix_dir)
    mkpath(matrix_dir)
end

# Préparer les données de base
data_path    = joinpath(current_dir, "data", case_file)
data         = PowerModels.parse_file(data_path)
formulations = [Chordal_AMD, Chordal_MFI, Chordal_MD]
merging_opts = [true, false]

# Préparer le CSV de résultats
results_csv = joinpath(current_dir,
    "results_" * replace(case_file, "." => "_") * num_suffix * ".csv")
header_df = DataFrame(Formulation=String[], Case=String[], Merge=Bool[], SolveTime=Float64[], Status=String[], Objective=Float64[])
CSV.write(results_csv, header_df)

# Boucle de lecture des fichiers JSON
for json_file in readdir(input_dir)
    if endswith(json_file, ".json")
        filepath = joinpath(input_dir, json_file)
        println("Reading file: ", filepath)
        loads = JSON.parsefile(filepath)

        # Mettre à jour les charges dans data
        for (id, load) in loads["load"]
            data["load"][id]["pd"] = load["pd"]
            data["load"][id]["qd"] = load["qd"]
        end

        for fm in formulations
            for merging in merging_opts
                try
                    println("Solving with formulation: ", fm, " merging: ", merging)
                    result, cadj1, _ = solve(data, fm, merging)

                    # Enregistrer les résultats dans le CSV
                    df = DataFrame(Formulation=string(fm), Case=json_file, Merge=merging,
                                   SolveTime=result["solve_time"], Status=string(result["termination_status"]),
                                   Objective=result["objective"])
                    CSV.write(results_csv, df, append=true)

                    # Générer un nom unique pour la matrice cadj1
                    timestamp  = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
                    safe_case  = replace(json_file, ".json" => "")
                    fm_name    = lowercase(string(fm))
                    merge_flag = merging ? "merged" : "nomerge"
                    mat_filename = joinpath(matrix_dir,
                        "cadj1_"
                        * safe_case * "_"
                        * fm_name   * "_"
                        * merge_flag * "_"
                        * timestamp  * ".jld2")

                    save_sparse_matrix(cadj1, mat_filename)
                catch e
                    println("Error solving with formulation: ", fm, " merging: ", merging)
                    println(e)
                    # Log erreur dans CSV
                    df_err = DataFrame(Formulation=string(fm), Case=json_file, Merge=merging,
                                       SolveTime=NaN, Status="error", Objective=NaN)
                    CSV.write(results_csv, df_err, append=true)
                end
                GC.gc()
            end
        end
        GC.gc()
    end
end
