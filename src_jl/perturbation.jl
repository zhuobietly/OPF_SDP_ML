module Perturbation

export perturbate!

function perturbate!(data, perturb_loads=(0,0))
    if perturb_loads[1] == 0
        return
    end
    Random.seed!(perturb_loads[2])
    sum_active = sum([load["pd"] for (bus, load) in data["load"]])
    sum_reactive = sum([load["qd"] for (bus, load) in data["load"]])
    for (bus, load) in data["load"]
        load["pd"] += randn() * abs(perturb_loads[1] * load["pd"])
        load["pd"] = max(0, load["pd"])
        load["qd"] += randn() * abs(perturb_loads[1] * load["qd"])
        load["qd"] = max(0, load["qd"])
    end
    println("Perturbation added to system.")
end

end # module