#!/usr/bin/env julia

# Analyze percentile distributions across trials for a participant.
# - For each trial, simulate J=500 with PDA/IBS fitted params
# - Filter sims by matching (c1, c2) to the observed trial
# - Compute percentiles of observed rt1/rt2 vs those matching sims
# - Plot percentile distributions and save a CSV summary

include("model.jl")
include("data.jl")
include("likelihood.jl")
include("pda.jl")

using JSON, CSV, DataFrames, Statistics, Random, Printf
using Plots

# ---------------- Configuration ----------------
Random.seed!(42)

const PARTICIPANT_ID = "wcb87506"             # <-- change participant here
const MODEL_NAME     = "model6"               # "model1" | "model5" | "model11"
const J              = 500                    # simulations per trial

# File paths
const DATA_FILE       = "data/Tree2_v3.json"
const PDA_RESULTS_FILE = "Tree2/results/model6_pda_20250829_142354.csv"
const IBS_RESULTS_FILE = "Tree2/results/model6_ibs_20250711_005050.csv"

# Output
const OUT_PREFIX = "participant_$(PARTICIPANT_ID)_$(MODEL_NAME)"
const CSV_OUT    = "$(OUT_PREFIX)_percentiles.csv"
const FIG_RT1    = "$(OUT_PREFIX)_percentiles_rt1.png"
const FIG_RT2    = "$(OUT_PREFIX)_percentiles_rt2.png"

# ---------------- Utilities ----------------
function load_participant_data(data_file::String, participant_id::String)
    trials = Trial[]
    open(data_file, "r") do io
        for line in eachline(io)
            s = strip(line); isempty(s) && continue
            t = JSON.parse(s)
            t["wid"] == participant_id || continue

            value1 = t["value1"]; value2 = t["value2"]
            rewards = [value1[1], value1[2], value2[1], value2[2], value2[3], value2[4]]

            push!(trials, Trial(
                rewards,
                t["choice1"],
                t["choice2"],
                t["rt1"],
                t["rt2"]
            ))
        end
    end
    isempty(trials) && error("Participant $participant_id not found in $data_file")
    return trials
end

function load_fitted_parameters(results_file::String, participant_id::String, model_name::String)
    df = CSV.read(results_file, DataFrame)
    row = df[df.wid .== participant_id, :]
    nrow(row) == 0 && error("Participant $participant_id not found in $results_file")

    if model_name == "model1"
        params = [
            row.d1[1],
            row.d2[1],
            row.θ1[1],
            row.θ2[1],
            row.T1[1],
            row.T2[1]
        ]
    elseif model_name == "model5"
        params = [
            row.d[1],
            row.θ1[1],
            row.θ2[1],
            row.T1[1],
            row.T2[1]
        ]
    elseif model_name == "model6"
        params = [
            row.d1[1],
            row.d2[1],
            row.θ1[1],
            row.θ2[1],
            row.T1[1],
            row.T2[1]
        ]
    elseif model_name == "model11"
        params = [
            row.d[1],
            row.θ0[1],
            row.vigor1[1],
            row.vigor2[1],
            row.T1[1],
            row.T2[1]
        ]
    else
        error("Model $model_name param extraction not implemented")
    end
    return params
end

function get_model_function(model_name::String)
    if model_name == "model1"
        return model1
    elseif model_name == "model5"
        return model5
    elseif model_name == "model6"
        return model6
    elseif model_name == "model11"
        return model11
    else
        error("Model function for $model_name not implemented")
    end
end

@inline function percentile_leq(samples::Vector{Float64}, x::Float64)
    # 100 * P(S ≤ x)
    isempty(samples) && return NaN
    return 100.0 * mean(samples .<= x)
end

# ---------------- Core routine ----------------
"""
    compute_percentiles_over_trials(trials, model_func, params; J=500)

Return a DataFrame with per-trial percentiles for RT1/RT2.
For each trial:
  - simulate J times,
  - filter simulated outcomes by matching (c1, c2),
  - compute percentile of observed rt1/rt2 vs matching simulated RTs.
"""
function compute_percentiles_over_trials(trials::Vector{Trial},
                                         model_func::Function,
                                         params::Vector{Float64};
                                         J::Int=J)
    N = length(trials)
    trial_idx     = Vector{Int}(undef, N)
    obs_c1        = Vector{Int}(undef, N)
    obs_c2        = Vector{Int}(undef, N)
    obs_rt1       = Vector{Float64}(undef, N)
    obs_rt2       = Vector{Float64}(undef, N)
    pct_rt1       = Vector{Union{Missing,Float64}}(missing, N)
    pct_rt2       = Vector{Union{Missing,Float64}}(missing, N)
    n_match_store = Vector{Int}(undef, N)

    for (i, tr) in enumerate(trials)
        trial_idx[i] = i
        obs_c1[i]    = tr.choice1
        obs_c2[i]    = tr.choice2
        obs_rt1[i]   = tr.rt1
        obs_rt2[i]   = tr.rt2

        # Simulate
        c1s, rt1s, c2s, rt2s = simulate_batch(model_func, params, tr.rewards, J)

        # Filter by matching (c1, c2)
        idxs = findall(j -> c1s[j] == tr.choice1 && c2s[j] == tr.choice2, 1:length(c1s))
        n_match = length(idxs)
        n_match_store[i] = n_match

        if n_match >= 1
            rt1_sim = rt1s[idxs]
            rt2_sim = rt2s[idxs]
            pct_rt1[i] = percentile_leq(rt1_sim, tr.rt1)
            pct_rt2[i] = percentile_leq(rt2_sim, tr.rt2)
        else
            pct_rt1[i] = missing
            pct_rt2[i] = missing
        end
    end

    return DataFrame(
        trial = trial_idx,
        c1 = obs_c1,
        c2 = obs_c2,
        rt1 = obs_rt1,
        rt2 = obs_rt2,
        percentile_rt1 = pct_rt1,
        percentile_rt2 = pct_rt2,
        n_matching = n_match_store,
    )
end

# ---------------- Plotting ----------------
function plot_percentile_histograms(df_pda::DataFrame, df_ibs::DataFrame; what::Symbol=:rt1, outfile::String="")
    # what = :rt1 or :rt2
    sym = (what == :rt1) ? :percentile_rt1 : :percentile_rt2
    pda_vals = collect(skipmissing(df_pda[!, sym]))
    ibs_vals = collect(skipmissing(df_ibs[!, sym]))

    if isempty(pda_vals) && isempty(ibs_vals)
        @warn "No percentile data to plot for $(String(what)). Skipping."
        return nothing
    end

    # Fixed bins 0..100 so the two histograms are comparable
    bins = collect(0:5:100)

    plt = plot(
        title = "Percentile Distribution ($(uppercase(String(what))))\nParticipant: $(PARTICIPANT_ID) · $(MODEL_NAME)",
        # xlabel = "Percentile (≤ observed)",
        xlabel = "Percentile",
        ylabel = "Count",
        xlim = (0, 100),
        legend = :top,
        size=(900, 500)
    )

    if !isempty(pda_vals)
        histogram!(plt, pda_vals; bins=bins, alpha=0.55, label="PDA", normalize=false)
        vline!(plt, [50.0]; color=:gray, lw=1, ls=:dash, label="")
    end
    if !isempty(ibs_vals)
        histogram!(plt, ibs_vals; bins=bins, alpha=0.55, label="IBS", normalize=false)
    end

    if !isempty(outfile)
        savefig(plt, outfile)
        @info "Saved $(outfile)"
    end
    return plt
end

# ---------------- Main ----------------
try
    println("="^80)
    println("Percentile Distribution Analysis")
    println("="^80)
    println("Participant: $(PARTICIPANT_ID)")
    println("Model: $(MODEL_NAME)")
    println("Simulations per trial: $(J)")

    # Load trials and model function
    trials     = load_participant_data(DATA_FILE, PARTICIPANT_ID)
    model_func = get_model_function(MODEL_NAME)
    println("Loaded $(length(trials)) trials for $(PARTICIPANT_ID)")

    # Load parameters
    pda_params = load_fitted_parameters(PDA_RESULTS_FILE, PARTICIPANT_ID, MODEL_NAME)
    ibs_params = load_fitted_parameters(IBS_RESULTS_FILE, PARTICIPANT_ID, MODEL_NAME)
    println("Loaded PDA/IBS fitted parameters")

    # Compute percentiles per trial
    println("\nComputing percentiles with PDA parameters...")
    df_pda = compute_percentiles_over_trials(trials, model_func, pda_params; J=J)
    println("  done. usable rows (non-missing): RT1=$(count(!ismissing, df_pda.percentile_rt1)), RT2=$(count(!ismissing, df_pda.percentile_rt2))")

    println("Computing percentiles with IBS parameters...")
    df_ibs = compute_percentiles_over_trials(trials, model_func, ibs_params; J=J)
    println("  done. usable rows (non-missing): RT1=$(count(!ismissing, df_ibs.percentile_rt1)), RT2=$(count(!ismissing, df_ibs.percentile_rt2))")

    # Merge and save CSV summary
    df_out = DataFrame(
        trial        = df_pda.trial,
        c1           = df_pda.c1,
        c2           = df_pda.c2,
        rt1          = df_pda.rt1,
        rt2          = df_pda.rt2,
        n_match_pda  = df_pda.n_matching,
        n_match_ibs  = df_ibs.n_matching,
        pda_pct_rt1  = df_pda.percentile_rt1,
        pda_pct_rt2  = df_pda.percentile_rt2,
        ibs_pct_rt1  = df_ibs.percentile_rt1,
        ibs_pct_rt2  = df_ibs.percentile_rt2,
    )
    CSV.write(CSV_OUT, df_out)
    println("\nSaved CSV: $(CSV_OUT)")

    # Plot percentile distributions
    plot_percentile_histograms(df_pda, df_ibs; what=:rt1, outfile=FIG_RT1)
    plot_percentile_histograms(df_pda, df_ibs; what=:rt2, outfile=FIG_RT2)

    println("\nSaved figures:")
    println("  - $(FIG_RT1)")
    println("  - $(FIG_RT2)")

    # Quick textual summary
    function quick_summary(name, v1, v2)
        v1c = collect(skipmissing(v1)); v2c = collect(skipmissing(v2))
        @printf("  %-3s RT1: n=%-3d mean=%6.2f median=%6.2f | RT2: n=%-3d mean=%6.2f median=%6.2f\n",
                name, length(v1c), mean(v1c), median(v1c),
                length(v2c), mean(v2c), median(v2c))
    end
    println("\nSummary of percentiles (in %):")
    quick_summary("PDA", df_pda.percentile_rt1, df_pda.percentile_rt2)
    quick_summary("IBS", df_ibs.percentile_rt1, df_ibs.percentile_rt2)

catch e
    println("❌ Analysis failed: $e")
    println("Troubleshooting:")
    println("  1) Check paths for DATA_FILE / PDA_RESULTS_FILE / IBS_RESULTS_FILE")
    println("  2) Verify participant id and MODEL_NAME columns exist")
    println("  3) Ensure simulate_batch(model, θ, rewards, J) returns (c1s, rt1s, c2s, rt2s)")
end