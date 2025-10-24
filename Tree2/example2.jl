#!/usr/bin/env julia
# --- single-stage DDM: synthesize (rt, choice), likelihood via DiffModels, fit with BADS ---

using Random
using Printf
using DiffModels
include("bads.jl")

# ----------------- First Passage Time Density -----------------
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
    rt::Float64      # decision time (dt=1ms)
    choice::Int      # 1 = upper, 0 = lower
end


function sample_trials(N::Int; μ::Float64=0.08, θ::Float64=1.0, σ::Float64=0.03,
                       tmax::Int=3000, rng::AbstractRNG=Random.default_rng())
    
    ts   = collect(1.0:1.0:Float64(tmax))
    pu   = [fpt_density_at(t; μ=μ, θ=θ, upper=true,  σ=σ)  for t in ts]
    pl   = [fpt_density_at(t; μ=μ, θ=θ, upper=false, σ=σ)  for t in ts]
    sumu = sum(pu); suml = sum(pl)
    
    p_upper = sumu / (sumu + suml + eps(Float64))
    pu_cond = (sumu > 0) ? (pu ./ sumu) : fill(0.0, length(ts))
    pl_cond = (suml > 0) ? (pl ./ suml) : fill(0.0, length(ts))

    # precompute CDFs for inverse transform
    cdfu = cumsum(pu_cond)
    cdfl = cumsum(pl_cond)

    trials = Vector{Trial}(undef, N)
    @inbounds for i in 1:N
        if rand(rng) < p_upper
            u = rand(rng)
            idx = searchsortedfirst(cdfu, u)
            rt  = ts[clamp(idx, 1, length(ts))]
            trials[i] = Trial(rt, 1)
        else
            u = rand(rng)
            idx = searchsortedfirst(cdfl, u)
            rt  = ts[clamp(idx, 1, length(ts))]
            trials[i] = Trial(rt, 0)
        end
    end
    return trials
end

# ----------------- Log-likelihood -----------------
@inline function loglik_trial(tr::Trial, θvec::Vector{Float64}; σ::Float64=0.01, eps::Float64=1e-12)
    μ, θ = θvec
    g = fpt_density_at(tr.rt; μ=μ, θ=θ, upper=(tr.choice==1), σ=σ)
    return log(max(g, eps))
end

# ----------------- Fit with BADS -----------------
function fit_ddm(trials::Vector{Trial};
                 σ::Float64=0.01,
                 x0::Vector{Float64} = [0.05, 1.0],     # [μ, θ]
                 lower_bounds::Vector{Float64} = [1e-4,  0.2],
                 upper_bounds::Vector{Float64} = [0.50,  3.0],
                 plausible_lower::Vector{Float64} = [0.01, 0.4],
                 plausible_upper::Vector{Float64} = [0.20, 2.0],
                 max_fun_evals::Int = 800)

    function objective_from_unit(x::Vector{Float64})
        f = -sum(tr -> loglik_trial(tr, x; σ=σ), trials)
        return f
    end

    bads_result = optimize_bads(objective_from_unit;
        x0 = x0,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds,
        plausible_lower_bounds = plausible_lower,
        plausible_upper_bounds = plausible_upper,
        max_fun_evals = max_fun_evals,
        uncertainty_handling = true
    )

    r    = get_result(bads_result)
    xopt = Float64.(r["x"])    
    fopt = Float64(r["fval"])
    return xopt, fopt, r
end

# ----------------- Demo -----------------
function demo()
    rng = MersenneTwister(123)

    # ground-truth
    μ_true  = 0.08
    θ_true  = 1.20
    σ       = 0.03

    println("Synthesizing data with μ=$(μ_true), θ=$(θ_true)")
    trials = sample_trials(800; μ=μ_true, θ=θ_true, σ=σ, tmax=2000, rng=rng)

    # fit
    xhat, fhat, info = fit_ddm(trials; σ=σ,
        x0=[0.04, 1.0],
        lower_bounds=[1e-4, 0.2],
        upper_bounds=[0.50, 3.0],
        plausible_lower=[0.02, 0.5],
        plausible_upper=[0.15, 2.0],
        max_fun_evals=1000
    )

    @printf("\nEstimated params: μ̂ = %.4f, θ̂ = %.4f\n", xhat[1], xhat[2])
    @printf("Negative log-likelihood at optimum: %.3f\n", fhat)
end

demo()

