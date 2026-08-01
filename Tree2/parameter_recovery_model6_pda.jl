#!/usr/bin/env julia
# Parameter recovery for full model 6 using PDA (Probability Density Approximation).
# 50 subjects, each with 100 trials randomly generated from model6.
# Fitting uses PDA likelihood and BADS optimizer.

using Distributed
using Dates
using Random
using Statistics
using DataFrames
using CSV

const SCRIPT_DIR = @__DIR__
cd(SCRIPT_DIR)

include("fitting.jl")

# Configuration
const N_SUBJECTS = 50
const N_TRIALS_PER_SUBJECT = 100
const MODEL_NAME = "model6"
const LIKELIHOOD_METHOD = "pda"
const OPTIMIZER = :BADS
const REWARD_MIN = 1
const REWARD_MAX = 5
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "results", "parameter_recovery")
const RANDOM_SEED = 20260130

# PDA-specific options
const PDA_J = 500
const PDA_MIN_SIMS = 1000
const PDA_MIN_MATCHING = 100
const PDA_MAX_SIMS = 10000
const PDA_EPS_FLOOR = 1e-16
const PDA_LAMBDA = 1.0
const PDA_KDE_MODE = :gaussian
const PDA_BW_RULE = :silverman

function sample_model6_parameters(rng::AbstractRNG=Random.GLOBAL_RNG)
    log_d_lo = log(1e-5)
    log_d_hi = log(1e-3)
    d1 = exp(log_d_lo + rand(rng) * (log_d_hi - log_d_lo))
    d2 = exp(log_d_lo + rand(rng) * (log_d_hi - log_d_lo))
    θ1 = 0.01 + rand(rng) * 0.99
    θ2 = 0.01 + rand(rng) * 0.99
    T1 = 100.0 + rand(rng) * 4900.0
    T2 = 100.0 + rand(rng) * 4900.0
    return [d1, d2, θ1, θ2, T1, T2]
end

@everywhere function _generate_trials_for_subject(wid::String, φ::Vector{Float64}, n_trials::Int;
                                                   reward_min::Int=1, reward_max::Int=5,
                                                   model_name::String="model6",
                                                   rng::AbstractRNG=Random.GLOBAL_RNG)
    config = get_model_config(model_name)
    model_func = config.model_function
    trials = Trial[]
    for _ in 1:n_trials
        rewards = Float64.([rand(rng, reward_min:reward_max) for _ in 1:6])
        result = model_func(φ, rewards)
        if result.timeout
            continue
        end
        path = rewards[3:6]
        push!(trials, Trial(wid, rewards, result.choice1, result.choice2,
                           Float64(result.rt1), Float64(result.rt2), path))
    end
    return trials
end

@everywhere function _run_single_recovery(subject_idx::Int, true_φ::Vector{Float64}, rng::AbstractRNG;
                                           n_trials_per_subject::Int=100,
                                           model_name::String="model6",
                                           likelihood_method::String="pda",
                                           optimizer::Symbol=:BADS,
                                           J::Int=500, min_sims::Int=1000, min_matching::Int=100,
                                           max_sims::Int=10000, eps_floor::Float64=1e-16,
                                           lambda::Float64=1.0, kde_mode::Symbol=:gaussian,
                                           bw_rule::Symbol=:silverman)
    wid = "recovery_$(lpad(subject_idx, 3, '0'))"
    trials = _generate_trials_for_subject(wid, true_φ, n_trials_per_subject; model_name=model_name, rng=rng)
    if length(trials) < 20
        @warn "Subject $wid: only $(length(trials)) valid trials (timeouts), skipping"
        return nothing
    end
    wid_fit, fitted_θ, negll, param_names, _ = fit_subject(
        wid, trials, model_name, likelihood_method;
        optimizer=optimizer,
        J=J, min_sims=min_sims, min_matching=min_matching, max_sims=max_sims,
        eps_floor=eps_floor, lambda=lambda, kde_mode=kde_mode, bw_rule=bw_rule
    )
    return (wid=wid, true_φ=true_φ, fitted_θ=fitted_θ, param_names=param_names,
            neglogl=negll, n_trials=length(trials))
end

function main()
    Random.seed!(RANDOM_SEED)

    println("="^70)
    println("Parameter recovery: Model 6, PDA likelihood")
    println("="^70)
    println("Subjects: $N_SUBJECTS")
    println("Trials per subject: $N_TRIALS_PER_SUBJECT")
    println("Model: $MODEL_NAME")
    println("Likelihood: $LIKELIHOOD_METHOD (J=$PDA_J)")
    println("Optimizer: $OPTIMIZER")
    println("Output dir: $OUTPUT_DIR")
    println("="^70)

    config = get_model_config(MODEL_NAME)
    param_names = config.param_names

    true_params_list = [sample_model6_parameters(MersenneTwister(RANDOM_SEED + i)) for i in 1:N_SUBJECTS]
    recovery_inputs = [(i, true_params_list[i], MersenneTwister(RANDOM_SEED + 1000 + i)) for i in 1:N_SUBJECTS]
    results_raw = pmap(recovery_inputs) do x
        _run_single_recovery(x[1], x[2], x[3];
                             n_trials_per_subject=N_TRIALS_PER_SUBJECT,
                             model_name=MODEL_NAME,
                             likelihood_method=LIKELIHOOD_METHOD,
                             optimizer=OPTIMIZER,
                             max_sims=PDA_MAX_SIMS, eps_floor=PDA_EPS_FLOOR, lambda=PDA_LAMBDA,
                             kde_mode=PDA_KDE_MODE, bw_rule=PDA_BW_RULE)
    end

    successful = filter(!isnothing, results_raw)
    println("\nSuccessful recoveries: $(length(successful)) / $N_SUBJECTS")

    if isempty(successful)
        @error "No successful recoveries."
        return nothing
    end

    n_params = length(param_names)
    col_names = [:wid; [Symbol("true_", p) for p in param_names];
                [Symbol("fitted_", p) for p in param_names]; :neglogl; :n_trials]
    df = DataFrame([T[] for T in [String; fill(Float64, 2 * n_params); Float64; Int]], col_names)

    for r in successful
        push!(df, [r.wid; r.true_φ; r.fitted_θ; r.neglogl; r.n_trials])
    end

    mkpath(OUTPUT_DIR)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_csv = joinpath(OUTPUT_DIR, "parameter_recovery_model6_pda_$(timestamp).csv")
    CSV.write(out_csv, df)
    println("Results saved to: $out_csv")

    return df
end

main()
