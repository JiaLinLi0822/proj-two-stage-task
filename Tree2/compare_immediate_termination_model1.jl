#!/usr/bin/env julia
# Read model1 RSS and PDA parameters, human data; run 10 simulations per trial
# on the whole dataset; report the proportion of trials with immediate termination
# (rt2 <= T2 + threshold) for RSS vs PDA. Usage (from Tree2/): julia compare_immediate_termination_model1.jl

using CSV
using DataFrames
using Printf
using Random

const SCRIPT_DIR = abspath(@__DIR__)
include(joinpath(SCRIPT_DIR, "model_configs.jl"))  # pulls in box.jl, model.jl
include(joinpath(SCRIPT_DIR, "data.jl"))
include(joinpath(SCRIPT_DIR, "add_info.jl"))

# Paths
const HUMAN_TRIAL_FILE = joinpath(SCRIPT_DIR, "data", "Tree2_v3.json")
const RSS_PARAM_FILE   = joinpath(SCRIPT_DIR, "results", "rss", "model1_RSS_de_20260129_133845.csv")
const PDA_PARAM_FILE   = joinpath(SCRIPT_DIR, "results", "pda", "model1_pda_BADS_20260125_215506.csv")
const OUT_CSV          = joinpath(SCRIPT_DIR, "results", "immediate_termination_model1_rss_pda.csv")

# Second-stage "immediate" = rt2 <= T2 + this (ms)
const IMMEDIATE_MS_THRESHOLD = 50.0
const N_SIMS_PER_TRIAL = 100
const RANDOM_SEED = 20260131

"""
Run n_sims simulations for one trial; return list of (diff2, immediate, timeout, choice_match).
immediate = second stage terminated in one step (rapid termination): rt2 <= T2 + threshold.
choice_match = (result.choice1, result.choice2) == (trial.choice1, trial.choice2).
"""
function simulate_trial_counts(trial, theta::Vector{Float64}, model_fun; n_sims::Int=N_SIMS_PER_TRIAL, threshold_ms::Float64=IMMEDIATE_MS_THRESHOLD)
    T2 = theta[6]
    rewards = trial.rewards
    value2 = [rewards[3], rewards[4], rewards[5], rewards[6]]
    out = Tuple{Float64, Bool, Bool, Bool}[]
    for _ in 1:n_sims
        result = model_fun(theta, rewards)
        diff2 = calculate_diff2(value2, result.choice1)
        immediate = result.rt2 <= T2 + threshold_ms
        choice_match = (result.choice1 == trial.choice1) && (result.choice2 == trial.choice2)
        push!(out, (diff2, immediate, false, choice_match))
    end
    return out
end

function main()
    Random.seed!(RANDOM_SEED)

    # 1. Load human trials
    if !isfile(HUMAN_TRIAL_FILE)
        error("Human trial file not found: $HUMAN_TRIAL_FILE")
    end
    trials = load_data(HUMAN_TRIAL_FILE)
    println("Loaded $(length(trials)) human trials from $HUMAN_TRIAL_FILE")

    # 2. Load model1 RSS (group) and PDA (per-subject) parameters
    if !isfile(RSS_PARAM_FILE)
        error("RSS parameter file not found: $RSS_PARAM_FILE")
    end
    theta_rss = load_rss_parameters(RSS_PARAM_FILE, "model1")
    println("Loaded RSS (group) parameters for model1 from $RSS_PARAM_FILE")

    if !isfile(PDA_PARAM_FILE)
        error("PDA parameter file not found: $PDA_PARAM_FILE")
    end
    pda_param_dict = load_fitted_parameters(PDA_PARAM_FILE, "model1")
    println("Loaded PDA (per-subject) parameters for $(length(pda_param_dict)) subjects from $PDA_PARAM_FILE")

    config = get_model_config("model1")
    model_fun = config.model_function

    # 3. Simulate: RSS (one theta for all trials), PDA (theta per subject)
    # Each record: (diff2, immediate, timeout, choice_match)
    rss_records = Tuple{Float64, Bool, Bool, Bool}[]
    pda_records = Tuple{Float64, Bool, Bool, Bool}[]
    n_trials_rss = 0
    n_trials_pda = 0
    for (idx, trial) in enumerate(trials)
        for t in simulate_trial_counts(trial, theta_rss, model_fun; n_sims=N_SIMS_PER_TRIAL)
            push!(rss_records, t)
        end
        n_trials_rss += 1
        wid = trial.wid
        if haskey(pda_param_dict, wid)
            for t in simulate_trial_counts(trial, pda_param_dict[wid], model_fun; n_sims=N_SIMS_PER_TRIAL)
                push!(pda_records, t)
            end
            n_trials_pda += 1
        end
        if idx % 500 == 0
            println("  Processed $idx / $(length(trials)) trials...")
        end
    end

    # 4. Count rapid termination only among simulations that match human choice pair (choice1, choice2)
    rss_matching = filter(r -> r[4], rss_records)
    rss_total_matching = length(rss_matching)
    rss_immediate = count(r -> r[2], rss_matching)
    rss_pct = rss_total_matching > 0 ? 100.0 * rss_immediate / rss_total_matching : 0.0

    pda_matching = filter(r -> r[4], pda_records)
    pda_total_matching = length(pda_matching)
    pda_immediate = count(r -> r[2], pda_matching)
    pda_pct = pda_total_matching > 0 ? 100.0 * pda_immediate / pda_total_matching : 0.0

    # 5. Report
    println("\n=== Rapid termination (rt2 <= T2 + $(IMMEDIATE_MS_THRESHOLD) ms) ===")
    println("  Only among simulations where (choice1, choice2) matches human data")
    println("  $N_SIMS_PER_TRIAL simulations per trial, whole dataset")
    println("  RSS: $rss_immediate / $rss_total_matching matching sims = $(round(rss_pct; digits=2))%")
    println("  PDA: $pda_immediate / $pda_total_matching matching sims = $(round(pda_pct; digits=2))%")

    # 6. Save summary to CSV
    mkpath(dirname(OUT_CSV))
    summary_df = DataFrame(
        method = ["RSS", "PDA"],
        n_trials = [n_trials_rss, n_trials_pda],
        n_sims_per_trial = [N_SIMS_PER_TRIAL, N_SIMS_PER_TRIAL],
        n_total_sims = [length(rss_records), length(pda_records)],
        n_matching_choice = [rss_total_matching, pda_total_matching],
        n_immediate = [rss_immediate, pda_immediate],
        pct_immediate = [rss_pct, pda_pct],
    )
    CSV.write(OUT_CSV, summary_df)
    println("\nSaved: $OUT_CSV")
end

main()
