using PowerModels            # 为了 parse_file 读取 .m
include("loaddata_gen.jl")
using .LoadProfileGen
case_name = "case118"
m_path = joinpath("/home/goatoine/Documents/Lanyue/data/raw_data/$(case_name).m")    
N = 20
outdir = joinpath("/home/goatoine/Documents/Lanyue/data/load_profiles/$(case_name)")
LoadProfileGen.create_load_profiles_from_matpower(m_path, N, outdir)
println("完成：输出目录 => ", outdir)
function count_files(dir::AbstractString)
    count(isfile, readdir(dir; join=true))
end

println(count_files("/home/goatoine/Documents/Lanyue/data/load_profiles/$(case_name)"))
