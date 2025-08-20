using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using CSV
using DataFrames
using Random
using Ipopt

case_dir = joinpath("/home/goatoine/Documents/Lanyue/data/data/raw_data")
case = [filename for filename in readdir(case_dir) if endswith(filename, ".m")]
df = DataFrame(
    case = String[],
    form = String[],
    objective_value = Float64[],
    solve_time = Float64[],
    status = String[],
    SolveTime = Float64[],
    SolutionStatus = String[]
)
CSV.write("Results_DC.csv", df)
for c in case
    case_path = joinpath(@__DIR__, "data", c)
    data = PowerModels.parse_file(case_path)
    results = solve_dc_opf(data, Mosek.Optimizer)
    println(results.keys)
    df_c = DataFrame(
        case = c,
        form = "DC",
        objective_value = results["objective"],
        solve_time = results["solve_time"],
        status = results["termination_status"],
        SolveTime = results["solve_time"],
        SolutionStatus = results["primal_status"]
    )
    # Append the results to the CSV file
    CSV.write("Results_DC.csv", df_c, append=true)

    results = solve_ac_opf(data, Ipopt.Optimizer)
    println(results.keys)
    df_c = DataFrame(
        case = c,
        form = "AC",
        objective_value = results["objective"],
        solve_time = results["solve_time"],
        status = results["termination_status"],
        SolveTime = results["solve_time"],
        SolutionStatus = results["primal_status"]
    )
    # Append the results to the CSV file
    CSV.write("Results_DC.csv", df_c, append=true)
end
