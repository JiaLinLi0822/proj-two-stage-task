#!/usr/bin/env julia
# Parameter recovery for full model 6 using RSS (Summary Statistics).
# 50 subjects, each with 100 trials randomly generated from model6.
# Fitting minimizes RSS between observed and model summary statistics; BADS optimizer.

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
const OPTIMIZER = :BADS
const REWARD_MIN = -4
const REWARD_MAX = 4
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "results", "parameter_recovery")
const RANDOM_SEED = 20260130
const MAX_FUN_EVALS = 1000

@everywhere begin

    subtree_vals(v2::AbstractVector{<:Real}, choice1::Int) = choice1 == 1 ? v2[1:2] : v2[3:4]

    function diff1(path::AbstractVector{<:Real})
        idx_max = argmax(path)
        others = [r for (i, r) in enumerate(path) if i != idx_max]
        return path[idx_max] - mean(others)
    end

    function diff2(v2::AbstractVector{<:Real}, choice1::Int)
        vals = subtree_vals(v2, choice1)
        return abs(vals[1] - vals[2])
    end

    function correct1(best_path_idx::Int, choice1::Int)
        return (best_path_idx ≤ 2 && choice1 == 1) || (best_path_idx ≥ 3 && choice1 == 2)
    end

    function correct2(v2::AbstractVector{<:Real}, choice1::Int, choice2::Int)
        c2_local = choice1 == 1 ? choice2 : choice2 - 2
        if c2_local < 1 || c2_local > length(subtree_vals(v2, choice1))
            return false
        end
        return subtree_vals(v2, choice1)[c2_local] == maximum(subtree_vals(v2, choice1))
    end

    function subtree_relation_code(path::AbstractVector{<:Real})
        idx_desc = sortperm(path; rev=true)
        best, second, third, worst = idx_desc
        subtree = i -> (i <= 2 ? 0 : 1)
        if subtree(best) == subtree(second)
            return 1
        elseif subtree(best) == subtree(third)
            return 2
        elseif subtree(best) == subtree(worst)
            return 3
        else
            return missing
        end
    end

    function add_info_rss(df::DataFrame)
        df.best_path_idx = map(path -> argmax(path), df.path)
        df.correct1 = map(correct1, df.best_path_idx, df.choice1)
        df.correct2 = map(correct2, df.value2, df.choice1, df.choice2)
        df.subtree_relation = map(subtree_relation_code, df.path)
        df.diff1 = map(diff1, df.path)
        df.diff2 = map(diff2, df.value2, df.choice1)
        return df
    end

    function per_subject_then_group_mean_rss(df::DataFrame; by_cols::Vector{Symbol}, y_col::Symbol)
        per_subj = combine(groupby(df, [:wid; by_cols...], sort=false), y_col => mean => :per_subject_mean)
        grp = combine(groupby(per_subj, by_cols, sort=false), :per_subject_mean => mean => :group_mean)
        return grp
    end

    function summary_stats_rss(df::DataFrame; scale_C=10_000.0, scale_D=1_000.0, scale_E=10_000.0)
        A_df = per_subject_then_group_mean_rss(df; by_cols=[:diff1], y_col=:correct1)
        B_df = per_subject_then_group_mean_rss(df; by_cols=[:diff2], y_col=:correct2)
        C_df = per_subject_then_group_mean_rss(df[df.correct1 .== true, :]; by_cols=[:diff1], y_col=:rt1)
        C_df.group_mean .= C_df.group_mean ./ scale_C
        D_df = per_subject_then_group_mean_rss(df[df.correct2 .== true, :]; by_cols=[:diff2], y_col=:rt2)
        D_df.group_mean .= D_df.group_mean ./ scale_D
        E_mask = (df.correct1 .== true) .& .!ismissing.(df.subtree_relation) .& in.(df.subtree_relation, Ref([1, 2, 3]))
        E_df = per_subject_then_group_mean_rss(df[E_mask, :]; by_cols=[:subtree_relation], y_col=:rt1)
        E_df.group_mean .= E_df.group_mean ./ scale_E
        return Dict{Symbol, DataFrame}(:A_df => A_df, :B_df => B_df, :C_df => C_df, :D_df => D_df, :E_df => E_df)
    end

    function _rss_between_rss(h::DataFrame, m::DataFrame, by_cols::Vector{Symbol})
        joined = outerjoin(h, m; on=by_cols, makeunique=true, matchmissing=:equal)
        hmean = coalesce.(joined.group_mean, 0.0)
        mmean = coalesce.(joined.group_mean_1, 0.0)
        return sum((mmean .- hmean) .^ 2)
    end

    function compute_rss_rss(Human_Dict::Dict{Symbol, DataFrame}, Model_Dict::Dict{Symbol, DataFrame})
        rss_A = _rss_between_rss(Human_Dict[:A_df], Model_Dict[:A_df], [:diff1])
        rss_B = _rss_between_rss(Human_Dict[:B_df], Model_Dict[:B_df], [:diff2])
        rss_C = _rss_between_rss(Human_Dict[:C_df], Model_Dict[:C_df], [:diff1])
        rss_D = _rss_between_rss(Human_Dict[:D_df], Model_Dict[:D_df], [:diff2])
        rss_E = _rss_between_rss(Human_Dict[:E_df], Model_Dict[:E_df], [:subtree_relation])
        return rss_A + rss_B + rss_C + rss_D + rss_E
    end

    function trials_to_observed_df_rss(trials)
        rows = Vector{NamedTuple}(undef, length(trials))
        @inbounds for i in eachindex(trials)
            tr = trials[i]
            rows[i] = (wid=tr.wid, rt1=tr.rt1, rt2=tr.rt2, choice1=tr.choice1, choice2=tr.choice2,
                      value1=tr.rewards[1:2], value2=tr.rewards[3:6], rewards=tr.rewards, path=tr.path)
        end
        return DataFrame(rows)
    end

    function simulate_trials_rss(trials, θ, config)
        model = Model(config.model_function, θ)
        rows = Vector{NamedTuple}(undef, length(trials))
        @inbounds for i in eachindex(trials)
            tr = trials[i]
            sim = simulate(model, tr)
            rows[i] = (wid=tr.wid, rt1=sim.rt1, rt2=sim.rt2, choice1=sim.choice1, choice2=sim.choice2,
                       value1=tr.rewards[1:2], value2=tr.rewards[3:6], rewards=tr.rewards, path=tr.path)
        end
        return DataFrame(rows)
    end

    function fit_subject_rss(wid::String, trials, model_name::String; max_fun_evals::Int=2000)
        config = get_model_config(model_name)
        box = config.hard_bounds
        param_names = collect(keys(box.dims))
        n = n_free(box)
        lbs = zeros(n)
        ubs = ones(n)
        pbox = config.plausible_bounds
        lower_dict = Dict(name => pbox.dims[name][1] for name in param_names)
        upper_dict = Dict(name => pbox.dims[name][2] for name in param_names)
        plbs = apply(box, lower_dict)
        pubs = apply(box, upper_dict)
        x0 = apply(box, config.initial_params)

        Human_df = trials_to_observed_df_rss(trials)
        Human_df = add_info_rss(Human_df)
        Human_Dict = summary_stats_rss(Human_df)

        function objective_function(x_unit)
            try
                x = Float64.(x_unit)
                θ_dict = box(x)
                θ = [θ_dict[name] for name in param_names]
                Model_df = simulate_trials_rss(trials, θ, config)
                Model_df = add_info_rss(Model_df)
                Model_Dict = summary_stats_rss(Model_df)
                return compute_rss_rss(Human_Dict, Model_Dict)
            catch e
                return 1e6
            end
        end

        try
            bads_result = optimize_bads(objective_function;
                x0=x0, lower_bounds=lbs, upper_bounds=ubs,
                plausible_lower_bounds=plbs, plausible_upper_bounds=pubs,
                max_fun_evals=max_fun_evals,
                uncertainty_handling=false, specify_target_noise=false)
            result_dict = get_result(bads_result)
            xopt_unit = Float64.(result_dict["x"])
            fopt = Float64(result_dict["fval"])
            real_xopt = box(xopt_unit)
            xopt = [real_xopt[name] for name in param_names]
            return (wid=wid, fitted_θ=xopt, rss=fopt, param_names=param_names, n_trials=length(trials))
        catch e
            @error "RSS fit failed for $wid: $e"
            x0_real = [config.initial_params[name] for name in param_names]
            return (wid=wid, fitted_θ=x0_real, rss=1e6, param_names=param_names, n_trials=length(trials))
        end
    end
end

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

@everywhere function _generate_trials_for_subject_rss(wid::String, φ::Vector{Float64}, n_trials::Int;
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

@everywhere function _run_single_recovery_rss(subject_idx::Int, true_φ::Vector{Float64}, rng::AbstractRNG;
                                                n_trials_per_subject::Int=100,
                                                model_name::String="model6",
                                                max_fun_evals::Int=2000)
    wid = "recovery_$(lpad(subject_idx, 3, '0'))"
    trials = _generate_trials_for_subject_rss(wid, true_φ, n_trials_per_subject; model_name=model_name, rng=rng)
    if length(trials) < 20
        @warn "Subject $wid: only $(length(trials)) valid trials (timeouts), skipping"
        return nothing
    end
    result = fit_subject_rss(wid, trials, model_name; max_fun_evals=max_fun_evals)
    return (wid=result.wid, true_φ=true_φ, fitted_θ=result.fitted_θ, param_names=result.param_names,
            rss=result.rss, n_trials=result.n_trials)
end

function main()
    Random.seed!(RANDOM_SEED)

    println("="^70)
    println("Parameter recovery: Model 6, RSS (summary statistics)")
    println("="^70)
    println("Subjects: $N_SUBJECTS")
    println("Trials per subject: $N_TRIALS_PER_SUBJECT")
    println("Model: $MODEL_NAME")
    println("Method: RSS + BADS")
    println("Output dir: $OUTPUT_DIR")
    println("="^70)

    config = get_model_config(MODEL_NAME)
    param_names = config.param_names

    true_params_list = [sample_model6_parameters(MersenneTwister(RANDOM_SEED + i)) for i in 1:N_SUBJECTS]
    recovery_inputs = [(i, true_params_list[i], MersenneTwister(RANDOM_SEED + 1000 + i)) for i in 1:N_SUBJECTS]
    results_raw = pmap(recovery_inputs) do x
        _run_single_recovery_rss(x[1], x[2], x[3];
                                 n_trials_per_subject=N_TRIALS_PER_SUBJECT,
                                 model_name=MODEL_NAME,
                                 max_fun_evals=MAX_FUN_EVALS)
    end

    successful = filter(!isnothing, results_raw)
    println("\nSuccessful recoveries: $(length(successful)) / $N_SUBJECTS")

    if isempty(successful)
        @error "No successful recoveries."
        return nothing
    end

    n_params = length(param_names)
    col_names = [:wid; [Symbol("true_", p) for p in param_names];
                [Symbol("fitted_", p) for p in param_names]; :rss; :n_trials]
    df = DataFrame([T[] for T in [String; fill(Float64, 2 * n_params); Float64; Int]], col_names)

    for r in successful
        push!(df, [r.wid; r.true_φ; r.fitted_θ; r.rss; r.n_trials])
    end

    mkpath(OUTPUT_DIR)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_csv = joinpath(OUTPUT_DIR, "parameter_recovery_model6_rss_$(timestamp).csv")
    CSV.write(out_csv, df)
    println("Results saved to: $out_csv")

    return df
end

main()
