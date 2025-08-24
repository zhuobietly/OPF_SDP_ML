module LightGC

export cleanup!

"""
cleanup!(vars...; do_gc=true)
- 将给定变量设为 `nothing`
- 可选触发一次 `GC.gc()`
"""
function cleanup!(vars...; do_gc::Bool=true)
    for v in vars
        try
            v[] = nothing   # 若传 Ref
        catch
            # 如果是普通对象，直接忽略
        end
    end
    do_gc && GC.gc()
end

"""
safe_close(figs...)
- 用于关闭绘图库对象 (PyPlot/Plots)
"""
function safe_close(figs...)
    for f in figs
        try
            close(f)
        catch
        end
    end
end

end # module
