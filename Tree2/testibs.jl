#!/usr/bin/env julia
# testibs.jl
# Compute PDA total log-likelihoods for each participant using IBS-fitted parameters.

using CSV
using DataFrames
using JSON
using Random
using Statistics
using Dates
using Printf

include("model.jl")
include("model_configs.jl")
include("data.jl")
include("likelihood.jl")
include("pda.jl")

# -----------------------
# Compute PDA total log-likelihood per participant
# -----------------------
function compute_pda_ll_per_participant(model_name::String;
        param_file::String="results/Tree2/model6_ibs_latest.csv",
        trial_file::String="data/Tree2_v3.json",
        output_csv::String="data/Tree2_sim/pda_ll_$(model_name)_ibs.csv",
        J::Int=1000,
        kde_mode::Symbol=:gaussian,
        bw_rule::Symbol=:silverman,
        logRT::Bool=true,
        eps_floor::Float64=1e-16,
        random_seed::Int=20250830)

    Random.seed!(random_seed)

    @assert haskey(MODEL_CONFIGS, model_name) "Unknown model '$(model_name)'. Available: $(join(string.(collect(keys(MODEL_CONFIGS))), ", "))"
    config = get_model_config(model_name)
    model_func = config.model_function

    println("="^70)
    println("PDA total log-likelihood | model=$model_name | J=$J | kde=$kde_mode | bw=$bw_rule | logRT=$logRT | eps=$eps_floor")
    println("="^70)

    println("Loading IBS-fitted parameters from: $param_file")
    param_dict = load_fitted_parameters(param_file, model_name)

    println("Loading trials from: $trial_file")
    trials_by_wid = load_data_by_subject(trial_file)

    # Iterate by participant
    rows = Vector{NamedTuple}()
    n_done = 0
    t0_global = time()

    for (wid, θ) in param_dict
        if !haskey(trials_by_wid, wid)
            # No trials for this wid in the JSON file — skip
            continue
        end

        trials = trials_by_wid[wid]
        tstart = time()
        # Guard against -Inf/NaN from any single trial by summing safely
        total_ll = 0.0
        n_trials = length(trials)

        # You can call the vectorized pda_loglike; we still keep a guard:
        total_ll = try
            pda_loglike(θ, trials, model_func; J=J, kde_mode=kde_mode, bw_rule=bw_rule, logRT=logRT, eps_floor=eps_floor)
        catch err
            @warn "pda_loglike failed for wid=$wid; falling back to per-trial accumulation" err=err
            s = 0.0
            for tr in trials
                ll = try
                    pda_loglike_single_trial(θ, tr, model_func; J=J, kde_mode=kde_mode, bw_rule=bw_rule, logRT=logRT, eps_floor=eps_floor)
                catch e2
                    @warn "single trial PDA failed; assigning -Inf" err=e2
                    -Inf
                end
                if isnan(ll)
                    @warn "NaN loglike; treating as -Inf" wid=wid
                    ll = -Inf
                end
                s += ll
            end
            s
        end

        runtime = time() - tstart

        push!(rows, (
            wid = wid,
            model = model_name,
            J = J,
            kde_mode = String(kde_mode),
            bw_rule = String(bw_rule),
            logRT = logRT,
            eps_floor = eps_floor,
            total_loglike = total_ll,
            num_trials = n_trials,
            runtime_sec = runtime,
            timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        ))

        n_done += 1
        if n_done % 5 == 0
            @info "Processed $n_done participants..."
        end
    end

    df = DataFrame(rows)
    mkpath(dirname(output_csv))
    CSV.write(output_csv, df)

    println("-"^70)
    @printf "Done. Participants written: %d | Output: %s\n" n_done output_csv
    @printf "Wall time: %.2f sec\n" (time() - t0_global)
    println("-"^70)

    return df
end



### Use example:
model_name = "model6"
param_file = "Tree2/results/model6_ibs_20250711_005050.csv"
trial_file = "data/Tree2_v3.json"
output_csv = "data/Tree2_sim/pda_ll_model6_ibs.csv"
compute_pda_ll_per_participant(model_name; param_file, trial_file, output_csv)