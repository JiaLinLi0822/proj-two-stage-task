using PyCall

@everywhere function optimize_bads(f::Function;
    x0::AbstractVector,
    lower_bounds::AbstractVector,
    upper_bounds::AbstractVector,
    plausible_lower_bounds::AbstractVector = lower_bounds,
    plausible_upper_bounds::AbstractVector = upper_bounds,
    max_fun_evals::Integer = 1000,
    uncertainty_handling::Bool = false)
    
    pybads = pyimport("pybads")

    py"""
    import numpy as _np

    _jl_callback = None

    def __set_jl_callback__(cb):
        global _jl_callback
        _jl_callback = cb

    def __py_obj__(x):
        x = _np.asarray(x, dtype=float).ravel().tolist()
        return float(_jl_callback(x))
    """

    pymain = pyimport("__main__")
    pymain.__set_jl_callback__(pyfunction(f, Vector{Float64}))

    BADS = pybads.BADS
    b = BADS(
        pymain.__py_obj__, collect(x0);
        lower_bounds           = collect(lower_bounds),
        upper_bounds           = collect(upper_bounds),
        plausible_lower_bounds = collect(plausible_lower_bounds),
        plausible_upper_bounds = collect(plausible_upper_bounds),
        options = Dict(
            "max_fun_evals"        => Int(max_fun_evals),
            "uncertainty_handling" => uncertainty_handling
        )
    )

    res = b.optimize()
    out = Dict{String,Any}()
    out["x"]         = Vector{Float64}(Array(res["x"]))
    out["fval"]      = Float64(res["fval"])
    out["exit_flag"] = Int(get(res, "exit_flag", 0))
    out["niters"]    = Int(get(res, "niters", 0))
    out["nevals"]    = Int(get(res, "nevals", 0))
    return out
end