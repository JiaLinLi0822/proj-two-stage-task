#!/usr/bin/env julia
# Compare analytical CDF with PDA empirical CDF for a specific trial
# Conduct Kolmogorov-Smirnov test

using DataFrames
using CSV
using Statistics
using Printf
using Dates
using LinearAlgebra
using DiffModels
using Random
using Plots
using HypothesisTests

include("data.jl")
include("pda.jl")
include("model6_config.jl")
include("model6.jl")
include("fpt.jl")
include("simulation.jl")
include("plot.jl")


# ----------------- Compute empirical CDF from PDA simulations -----------------
function empirical_cdf(data::Vector{Float64}, t_values::Vector{Float64})
    ecdf_vals = Float64[]
    for t in t_values
        p = mean(data .<= t)
        push!(ecdf_vals, p)
    end
    return ecdf_vals
end

# ----------------- Kolmogorov-Smirnov test -----------------
function perform_ks_test(analytical_cdf::Vector{Float64}, empirical_cdf::Vector{Float64}, t_values::Vector{Float64}, n_samples::Int)
    
    # Find the maximum difference
    diffs = abs.(analytical_cdf .- empirical_cdf)
    max_diff = maximum(diffs)
    max_diff_idx = argmax(diffs)
    max_diff_t = t_values[max_diff_idx]
    
    ks_statistic = max_diff * sqrt(n_samples)
    
    p_val = 0.0
    for k in 1:100
        p_val += 2 * (-1)^(k-1) * exp(-2 * k^2 * ks_statistic^2)
    end
    p_val = max(p_val, 0.0)
    
    return (ks_statistic=ks_statistic, p_value=p_val, max_diff=max_diff, max_diff_t=max_diff_t)
end

function compare_cdf_ks_test(wid::String, trial_idx::Int;
                            analytical_file::String = "results/analytical/model6_test_20250919_151351.csv",
                            data_file::String = "pda/data/Tree2_v3.json",
                            n_sims::Int = 10000,
                            n_time_points::Int = 500,
                            log_scale::Bool = false,
                            bw_scale::Float64 = 1.0,
                            use_adaptive::Bool = false,
                            adaptive_J::Int = 1000,
                            max_sims::Int = 1000000)
    
    println("="^70)
    println("Comparing analytical CDF vs PDA ECDF")
    println("Participant: $wid, Trial: $trial_idx")
    println("="^70)
    
    # [1] Load data ---------------------------------------------------------
    println("\n[1] Loading data...")
    subject_trials = load_data_by_subject(data_file)
    
    if !haskey(subject_trials, wid)
        error("Participant $wid not found in data!")
    end
    
    trials = subject_trials[wid]
    if trial_idx < 1 || trial_idx > length(trials)
        error("Trial index $trial_idx out of range [1, $(length(trials))]")
    end
    
    target_trial = trials[trial_idx]
    println("    Target trial:")
    println("      Choice1=$(target_trial.choice1), Choice2=$(target_trial.choice2)")
    println("      RT1=$(target_trial.rt1), RT2=$(target_trial.rt2)")
    println("      Rewards=$(target_trial.rewards)")
    
    # [2] Load analytical params -------------------------------------------
    println("\n[2] Loading analytical parameters...")
    if !isfile(analytical_file)
        error("Analytical file not found: $analytical_file")
    end
    analytical_df = CSV.read(analytical_file, DataFrame)
    analytical_row = filter(row -> row.wid == wid, analytical_df)
    
    if nrow(analytical_row) == 0
        error("Participant $wid not found in analytical results!")
    end
    analytical_params = [analytical_row[1, :d1], analytical_row[1, :d2],
                         analytical_row[1, :θ1], analytical_row[1, :θ2],
                         analytical_row[1, :T1], analytical_row[1, :T2]]
    println("    Parameters: d1=$(analytical_params[1]), d2=$(analytical_params[2]), θ1=$(analytical_params[3]), θ2=$(analytical_params[4]), T1=$(analytical_params[5]), T2=$(analytical_params[6])")
    
    # [3] Generate PDA samples ---------------------------------------------
    println("\n[3] Generating PDA samples...")
    if use_adaptive
        println("    Using adaptive sampling: J=$adaptive_J, max_sims=$max_sims")
        all_samples = pda_sampler(target_trial, analytical_params; J=adaptive_J, max_sims=max_sims)
    else
        println("    Using fixed sampling: n_sims=$n_sims per trial")
        all_samples = simulate_trials([target_trial], analytical_params; n_sims=n_sims)
    end
    
    # For empirical ECDF, still用 matching pair（条件在 target_pair 上）
    target_pair = (target_trial.choice1, target_trial.choice2)
    matching_sims = filter(r -> (r.choice1, r.choice2) == target_pair, all_samples)
    
    if length(matching_sims) < 2
        error("Insufficient matching samples: $(length(matching_sims)) (need at least 2)")
    end
    
    rt1_sim = [r.rt1 for r in matching_sims]
    rt2_sim = [r.rt2 for r in matching_sims]
    
    total_sims = length(all_samples)
    acceptance_rate = length(matching_sims) / total_sims
    
    println("    Generated $(length(matching_sims)) matching samples (out of $total_sims total)")
    println("    Acceptance rate: $(round(acceptance_rate, digits=4))")
    println("    RT1 range: [$(minimum(rt1_sim)), $(maximum(rt1_sim))]")
    println("    RT2 range: [$(minimum(rt2_sim)), $(maximum(rt2_sim))]")
    
    # [4] Time grid ---------------------------------------------------------
    t_min = min(minimum(rt1_sim), minimum(rt2_sim)) - 100
    t_max = max(maximum(rt1_sim), maximum(rt2_sim)) + 100
    t_values = collect(range(t_min, stop=t_max, length=n_time_points))
    
    # [4] Analytical CDF ----------------------------------------------------
    println("\n[4] Computing analytical CDFs...")
    cdf_rt1_analytical, cdf_rt2_analytical = analytical_cdf(target_trial, analytical_params; t_values=t_values)
    
    # Empirical ECDF
    println("    Computing empirical CDFs...")
    cdf_rt1_empirical = empirical_cdf(rt1_sim, t_values)
    cdf_rt2_empirical = empirical_cdf(rt2_sim, t_values)
    
    # [5] Analytical PDF ----------------------------------------------------
    println("\n[5] Computing analytical PDFs...")
    pdf_rt1_analytical, pdf_rt2_analytical = analytical_pdf(target_trial, analytical_params; t_values=t_values)
    
    # [6] PDA marginal PDFs/CDFs ----------------------------------
    println("\n[6] Computing PDA marginal PDFs and CDFs...")
    pda_cond = compute_pda_joint_choice_rt(all_samples, target_trial, t_values;
                                            logRT=log_scale,
                                            bw_scale=bw_scale,
                                            min_samples=10)
    
    pdf_rt1_pda = pda_cond.pdf_rt1
    cdf_rt1_pda = pda_cond.cdf_rt1
    pdf_rt2_pda = pda_cond.pdf_rt2
    cdf_rt2_pda = pda_cond.cdf_rt2
    
    # [7] KS tests ----------------------------------------------------------
    println("\n[7] Performing Kolmogorov-Smirnov tests...")
    
    ks_rt1_emp = perform_ks_test(cdf_rt1_analytical, cdf_rt1_empirical, t_values, length(rt1_sim))
    ks_rt2_emp = perform_ks_test(cdf_rt2_analytical, cdf_rt2_empirical, t_values, length(rt2_sim))
    
    println("    Analytical vs Empirical:")
    println("      RT1: D=$(round(ks_rt1_emp.max_diff, digits=4)), p=$(round(ks_rt1_emp.p_value, digits=4)), at t=$(round(ks_rt1_emp.max_diff_t, digits=1))")
    println("      RT2: D=$(round(ks_rt2_emp.max_diff, digits=4)), p=$(round(ks_rt2_emp.p_value, digits=4)), at t=$(round(ks_rt2_emp.max_diff_t, digits=1))")
    
    # analytical vs PDA marginal
    ks_rt1_pda = perform_ks_test(cdf_rt1_analytical, cdf_rt1_pda, t_values, length(all_samples))
    ks_rt2_pda = perform_ks_test(cdf_rt2_analytical, cdf_rt2_pda, t_values, length(all_samples))
    
    println("    Analytical vs PDA marginal:")
    println("      RT1: D=$(round(ks_rt1_pda.max_diff, digits=4)), p=$(round(ks_rt1_pda.p_value, digits=4)), at t=$(round(ks_rt1_pda.max_diff_t, digits=1))")
    println("      RT2: D=$(round(ks_rt2_pda.max_diff, digits=4)), p=$(round(ks_rt2_pda.p_value, digits=4)), at t=$(round(ks_rt2_pda.max_diff_t, digits=1))")
    
    # [8] Plots -------------------------------------------------------------
    println("\n[8] Creating plots...")
    
    plt = plot_cdf_pdf_comparison(wid, trial_idx,
                                   t_values,
                                   rt1_sim, rt2_sim,
                                   pdf_rt1_pda, pdf_rt2_pda,             
                                   pdf_rt1_analytical, pdf_rt2_analytical,
                                   cdf_rt1_analytical, cdf_rt2_analytical,
                                   cdf_rt1_empirical, cdf_rt2_empirical,
                                   cdf_rt1_pda, cdf_rt2_pda,
                                   target_trial,
                                   ks_rt1_emp, ks_rt2_emp;
                                   log_scale=log_scale,
                                   output_file="pda/figures/$(wid)_trial$(trial_idx)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).png")
    
    return plt
end

# ----------------- Run comparison -----------------
wid = "wdaebe9a"
trial_idx =  92
log_scale = false
bw_scale = 0.8
use_adaptive = false
adaptive_J = 1000
max_sims = 10000

analytical_file = "pda/results/model6_test_20250919_151351.csv"

results = compare_cdf_ks_test(wid, trial_idx; 
                              analytical_file=analytical_file, 
                              n_sims=10000, 
                              log_scale=log_scale, 
                              bw_scale=bw_scale,
                              use_adaptive=use_adaptive,
                              adaptive_J=adaptive_J,
                              max_sims=max_sims)

