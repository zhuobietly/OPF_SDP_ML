module PMWarnFilter
using Logging

"""
只屏蔽 PowerModels 打印的“this code only supports angmin/angmax ... tightening ...”类警告；
其他日志（Info/Debug/Error）与求解器输出照常。
"""
struct NoPMAngleWarnsLogger <: AbstractLogger
    parent::AbstractLogger
end

# 把阈值/异常捕获/过滤策略都交给父 logger，保证其余日志行为不变
Logging.min_enabled_level(l::NoPMAngleWarnsLogger) = Logging.min_enabled_level(l.parent)
Logging.catch_exceptions(::NoPMAngleWarnsLogger) = true
Logging.shouldlog(l::NoPMAngleWarnsLogger, level, _module, group, id) =
    Logging.shouldlog(l.parent, level, _module, group, id)

function Logging.handle_message(l::NoPMAngleWarnsLogger, level, message,
        _module, group, id, file, line; kwargs...)
    if level == Logging.Warn
        # 取出消息文本
        msg = try
            message isa AbstractString ? message : string(message)
        catch
            sprint(show, message)
        end
        # 命中 PowerModels 的角度收紧提示 => 吞掉
        if occursin("this code only supports angmin", msg) ||
           occursin("this code only supports angmax", msg)
            return
        end
    end
    # 其它情况原样转发给父 logger
    Logging.handle_message(l.parent, level, message, _module, group, id, file, line; kwargs...)
end

end # module
