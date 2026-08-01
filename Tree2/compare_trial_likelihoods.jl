#!/usr/bin/env julia
# Compare trial-by-trial likelihoods between PDA and analytical methods

using DataFrames
using CSV
using Statistics
using Printf
using Dates
using LinearAlgebra
using DiffModels
using Random
using Plots

include("data.jl")
include("pda.jl")
include("model_configs.jl")
include("model.jl")
include("fpt.jl")
include("plot.jl")

const SCRIPT_DIR = @__DIR__
const DEFAULT_DATA_FILE = joinpath(SCRIPT_DIR, "data", "Tree2_v3.json")


function loglik_trial_pda(tr::Trial, params::Vector{Float64}, model_function::Function;
                                        J::Int=1000,
                                        min_sims::Int=1000,
                                        min_matching::Int=100,
                                        eps_floor::Float64=1e-16,
                                        bw_scale::Float64=1.0,
                                        max_sims::Int=1000000,
                                        logRT::Bool=true)

    samples = pda_sampler(tr, params, model_function; 
                         min_sims=max(min_sims, J),
                         min_matching=min_matching,
                         max_sims=max_sims)

    pairs = [(1, 1), (1, 2), (2, 3), (2, 4)]
    spdf = build_mixed2d_spdf(samples, tr;
                                pairs=pairs,
                                bw_rule=:silverman,
                                logRT=logRT,
                                eps_floor=eps_floor,
                                bw_scale=bw_scale)
    
    loglik, used_eps = mixed2d_logpdf(spdf, tr, 1.0)
    
    return loglik, used_eps, length(samples)
end


function compare_likelihoods(wid::String;
                            params_file::String = joinpath(SCRIPT_DIR, "results", "pda", "model6_pda_BADS_20251024_151315.csv"),
                            data_file::String = DEFAULT_DATA_FILE,
                            output_file::Union{String,Nothing} = nothing,
                            model_name::String = "model6",
                            J::Int = 1000,
                            min_sims::Int = 1000,
                            min_matching::Int = 100,
                            max_sims::Int = 1000000,
                            eps_floor::Float64 = 1e-16,
                            bw_scale::Float64 = 1.0,
                            logRT::Bool = true)
    
    println("="^70)
    println("Comparing trial-by-trial likelihoods for participant: $wid")
    println("="^70)
    
    println("\n[1] Loading data...")
    subject_trials = load_data_by_subject(data_file)
    
    if !haskey(subject_trials, wid)
        error("Participant $wid not found in data!")
    end
    
    trials = subject_trials[wid]
    n_trials = length(trials)
    println("    Found $n_trials trials for $wid")
    
    println("\n[2] Loading PDA parameters...")
    
    if !isfile(params_file)
        error("Parameters file not found: $params_file")
    end
    params_df = CSV.read(params_file, DataFrame)
    params_row = filter(row -> row.wid == wid, params_df)
    if nrow(params_row) == 0
        error("Participant $wid not found in parameters file!")
    end
    params_params = [params_row[1, :d1], params_row[1, :d2], params_row[1, :θ1], 
                  params_row[1, :θ2], params_row[1, :T1], params_row[1, :T2]]
    println("    Parameters: d1=$(params_params[1]), d2=$(params_params[2]), θ1=$(params_params[3]), θ2=$(params_params[4]), T1=$(params_params[5]), T2=$(params_params[6])")
    
    config = get_model_config(model_name)
    model_function = config.model_function
    println("    Using model: $model_name")
    
    results_data = []
    
    for (idx, trial) in enumerate(trials)
        
        # PDA likelihood
        pda_loglik, pda_used_eps, pda_n_samples = loglik_trial_pda(trial, params_params, model_function;
                                                                J=J, min_sims=min_sims, min_matching=min_matching,
                                                                eps_floor=eps_floor,
                                                                bw_scale=bw_scale, max_sims=max_sims, logRT=logRT)
        
        # Analytical likelihood
        analytical_loglik = loglik_trial_stagewise(trial, params_params)
        
        diff2_val = missing
        if hasproperty(trial, :diff2)
            diff2_val = getproperty(trial, :diff2)
        elseif hasproperty(trial, :abs_reward_diff2)
            diff2_val = getproperty(trial, :abs_reward_diff2)
        elseif hasproperty(trial, :reward_diff2)
            diff2_val = abs(getproperty(trial, :reward_diff2))
        end

        loglik_diff = pda_loglik - analytical_loglik

        push!(results_data, (
            trial_idx = idx,
            choice1 = trial.choice1,
            choice2 = trial.choice2,
            rt1 = trial.rt1,
            rt2 = trial.rt2,
            diff2 = diff2_val,

            pda_loglik = pda_loglik,
            pda_used_eps = pda_used_eps,
            pda_n_samples = pda_n_samples,

            analytical_loglik = analytical_loglik,

            loglik_diff = loglik_diff,
            abs_err = abs(loglik_diff),
            mean_ll = 0.5 * (pda_loglik + analytical_loglik)
        ))
        
        if idx % 10 == 0
            println("    Processed $idx/$n_trials trials... ")
        end
    end
    
    
    df = DataFrame(results_data)

    if output_file === nothing
        output_file = "pda/results/$(wid)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    end
    
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    CSV.write(output_file, df)
    println("\n[3] Results saved to: $output_file")
    
    return df, params_params
end

# ----------------- Run comparison -----------------
wid = "w6eb2a0a"  
params_file = joinpath(@__DIR__, "results", "pda","model6_pda_BADS_20260126_171314.csv")

df, params_params = compare_likelihoods(wid;
                        params_file=params_file,
                        model_name="model6",
                        J=1000,
                        min_sims=1000,
                        min_matching=1000,
                        max_sims=10000,
                        eps_floor=1e-64,
                        bw_scale=1.0,
                        logRT=true)



