#!/usr/bin/env julia
# Parameter recovery for full model 6 using analytical likelihood.
# 50 subjects, each with 100 trials randomly generated from model6.
# Fitting uses analytical (FPT) likelihood and BADS optimizer.

using Distributed
using Dates
using Random
using Statistics
using DataFrames
using CSV

const SCRIPT_DIR = @__DIR__
cd(SCRIPT_DIR)

include("fitting.jl")

const N_SUBJECTS = 50
const N_TRIALS_PER_SUBJECT = 100
const MODEL_NAME = "model6"
const LIKELIHOOD_METHOD = "analytical"
const OPTIMIZER = :BADS
const REWARD_MIN = -4
const REWARD_MAX = 4
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "results", "parameter_recovery")
const RANDOM_SEED = 20260130

function sample_model6_parameters(rng::AbstractRNG=Random.GLOBAL_RNG)
    # Plausible bounds from model_configs: d1,d2 log [1e-5, 1e-3], θ1,θ2 [0.01, 1.0], T1,T2 [100, 5000]
    log_d_lo = log(1e-5)
    log_d_hi = log(1e-3)
    d1 = exp(log_d_lo + rand(rng) * (log_d_hi - log_d_lo))
    d2 = exp(log_d_lo + rand(rng) * (log_d_hi - log_d_lo))
    θ1 = 0.01 + rand(rng) * 0.99   # [0.01, 1.0]
    θ2 = 0.01 + rand(rng) * 0.99
    T1 = 100.0 + rand(rng) * 4900.0   # [100, 5000]
    T2 = 100.0 + rand(rng) * 4900.0
    return [d1, d2, θ1, θ2, T1, T2]
end

@everywhere function _generate_trials_for_subject(wid::String, φ::Vector{Float64}, n_trials::Int;
                                                   reward_min::Int=-4, reward_max::Int=4,
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
                                           likelihood_method::String="analytical",
                                           optimizer::Symbol=:BADS)
    wid = "recovery_$(lpad(subject_idx, 3, '0'))"
    trials = _generate_trials_for_subject(wid, true_φ, n_trials_per_subject; model_name=model_name, rng=rng)
    if length(trials) < 20
        @warn "Subject $wid: only $(length(trials)) valid trials (timeouts), skipping"
        return nothing
    end
    wid_fit, fitted_θ, negll, param_names, _ = fit_subject(
        wid, trials, model_name, likelihood_method;
        optimizer=optimizer
    )
    return (wid=wid, true_φ=true_φ, fitted_θ=fitted_θ, param_names=param_names,
            neglogl=negll, n_trials=length(trials))
end

function main()
    Random.seed!(RANDOM_SEED)

    println("="^70)
    println("Parameter recovery: Model 6, analytical likelihood")
    println("="^70)
    println("Subjects: $N_SUBJECTS")
    println("Trials per subject: $N_TRIALS_PER_SUBJECT")
    println("Model: $MODEL_NAME")
    println("Likelihood: $LIKELIHOOD_METHOD (FPT)")
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
                             optimizer=OPTIMIZER)
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
        row = [r.wid;
               r.true_φ;
               r.fitted_θ;
               r.neglogl;
               r.n_trials]
        push!(df, row)
    end

    mkpath(OUTPUT_DIR)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_csv = joinpath(OUTPUT_DIR, "parameter_recovery_model6_analytical_$(timestamp).csv")
    CSV.write(out_csv, df)
    println("Results saved to: $out_csv")

    return df
end

main()
