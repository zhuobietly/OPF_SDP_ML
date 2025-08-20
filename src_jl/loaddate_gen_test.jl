using PowerModels            # 为了 parse_file 读取 .m
include("loaddata_gen.jl")
using .LoadProfileGen
m_path = joinpath("/home/goatoine/Documents/Lanyue/data/raw_data/case2746wop.m")    
N = 3
outdir = joinpath("/home/goatoine/Documents/Lanyue/data/load_profiles/case2746wop")
LoadProfileGen.create_load_profiles_from_matpower(m_path, N, outdir)
println("完成：输出目录 => ", outdir)
function count_files(dir::AbstractString)
    count(isfile, readdir(dir; join=true))
end

println(count_files("/home/goatoine/Documents/Lanyue/data/load_profiles/case2746wop"))
