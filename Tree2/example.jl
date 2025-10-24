#!/usr/bin/env julia

using Random
using Printf
using DiffModels
using PyCall

function optimize_bads(f::Function;
    x0::AbstractVector,
    lower_bounds::AbstractVector,
    upper_bounds::AbstractVector,
    plausible_lower_bounds::AbstractVector = lower_bounds,
    plausible_upper_bounds::AbstractVector = upper_bounds,
    max_fun_evals::Integer = 1000,
    uncertainty_handling::Bool = false
)
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

@inline function fpt_density_at(t::Float64; μ::Float64, θ::Float64, upper::Bool, σ::Float64=0.01)
    if t <= 0.0
        return 0.0
    end
    μs = μ/σ
    θs = θ/σ
    dt = 1
    d  = ConstDrift(μs, dt)
    B  = ConstSymBounds(θs, dt)
    return upper ? pdfu(d, B, t) : pdfl(d, B, t)
end

struct Trial
    rt::Float64      # decision time (dt=1)
    choice::Int      # 1 = upper, 0 = lower
end

function sample_trials(N::Int; μ::Float64=0.08, θ::Float64=1.0, σ::Float64=0.03,
                       tmax::Int=2000, rng::AbstractRNG=Random.default_rng())
    
    ts = collect(1.0:1.0:Float64(tmax))
    pu = [fpt_density_at(t; μ=μ, θ=θ, upper=true,  σ=σ) for t in ts]
    pl = [fpt_density_at(t; μ=μ, θ=θ, upper=false, σ=σ) for t in ts]
    sumu, suml = sum(pu), sum(pl)

    p_upper = sumu / (sumu + suml + eps(Float64))
    pu_cond = (sumu > 0) ? (pu ./ sumu) : fill(0.0, length(ts))
    pl_cond = (suml > 0) ? (pl ./ suml) : fill(0.0, length(ts))
    cdfu, cdfl = cumsum(pu_cond), cumsum(pl_cond)

    trials = Vector{Trial}(undef, N)
    @inbounds for i in 1:N
        if rand(rng) < p_upper
            idx = searchsortedfirst(cdfu, rand(rng))
            trials[i] = Trial(ts[clamp(idx, 1, length(ts))], 1)
        else
            idx = searchsortedfirst(cdfl, rand(rng))
            trials[i] = Trial(ts[clamp(idx, 1, length(ts))], 0)
        end
    end
    return trials
end

@inline function loglik_trial(tr::Trial, θvec::Vector{Float64}; σ::Float64=0.01, eps::Float64=1e-64)
    μ, θ = θvec
    g = fpt_density_at(tr.rt; μ=μ, θ=θ, upper=(tr.choice==1), σ=σ)
    return log(max(g, eps))
end

function fit_ddm(trials::Vector{Trial};
                σ::Float64=0.01,
                x0::Vector{Float64} = [0.05, 1.0],
                lower_bounds::Vector{Float64} = [1e-4,  0.2],
                upper_bounds::Vector{Float64} = [0.50,  3.0],
                plausible_lower::Vector{Float64} = [0.01, 0.4],
                plausible_upper::Vector{Float64} = [0.20, 2.0],
                max_fun_evals::Int = 1000,
                uncertainty_handling::Bool = false)
    
    function objective(x::Vector{Float64})
        -sum(tr -> loglik_trial(tr, x; σ=σ), trials)
    end

    r = optimize_bads(objective;
        x0 = x0,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds,
        plausible_lower_bounds = plausible_lower,
        plausible_upper_bounds = plausible_upper,
        max_fun_evals = max_fun_evals,
        uncertainty_handling = uncertainty_handling
    )
    xopt = Float64.(r["x"])
    fopt = Float64(r["fval"])
    return xopt, fopt, r
end

function demo()
    rng = MersenneTwister(123)

    μ_true = 0.08
    θ_true = 1.20
    σ = 0.05

    println("Synthesizing data with μ=$(μ_true), θ=$(θ_true)")
    trials = sample_trials(1000; μ=μ_true, θ=θ_true, σ=σ, tmax=3000, rng=rng)

    xhat, fhat, info = fit_ddm(trials; σ=σ,
        x0=[0.06, 1.6],
        lower_bounds=[1e-4, 0.2],
        upper_bounds=[0.50, 5.0],
        plausible_lower=[0.02, 0.5],
        plausible_upper=[0.15, 4.0],
        max_fun_evals=1000,
        uncertainty_handling=true
    )

    @printf("\nEstimated params: μ̂ = %.4f, θ̂ = %.4f\n", xhat[1], xhat[2])
    @printf("Negative log-likelihood at optimum: %.3f\n", fhat)
end

demo()