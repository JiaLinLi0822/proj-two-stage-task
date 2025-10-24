#!/usr/bin/env julia
using Distributed
using Logging
using Statistics
using JSON

@everywhere begin
    using LinearAlgebra
    using DiffModels
    using Random, Printf
    using Dates
    using DataFrames
    using CSV
    using Statistics
    using Logging
    using BlackBoxOptim
    LinearAlgebra.BLAS.set_num_threads(1)
    
    include("data.jl")
    include("bads.jl")
    include("model_configs.jl")
    
    Logging.disable_logging(Logging.Warn)
end

# ----------------- Helper functions -----------------
subtree_vals(v2::AbstractVector{<:Real}, choice1::Int) = choice1 == 1 ? v2[1:2] : v2[3:4]

function diff1(path::AbstractVector{<:Real})
    idx_max = argmax(path)
    others = [r for (i,r) in enumerate(path) if i != idx_max]
    return path[idx_max] - mean(others)
end

function diff2(v2::AbstractVector{<:Real}, choice1::Int)
    vals = subtree_vals(v2, choice1)
    return abs(vals[1] - vals[2])
end

function correct1(best_path_idx::Int, choice1::Int)
    return (best_path_idx ≤ 2) && (choice1 == 1) || ((best_path_idx ≥ 3) && (choice1 == 2))
end

function correct2(v2::AbstractVector{<:Real}, choice1::Int, choice2::Int)
    c2_local = choice1 == 1 ? choice2 : choice2 - 2
    # Ensure c2_local is within valid range
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

function add_info(df::DataFrame)

    df.best_path_idx = map(path -> argmax(path), df.path)

    df.correct1 = map(correct1, df.best_path_idx, df.choice1)
    df.correct2 = map(correct2, df.value2, df.choice1, df.choice2)
    df.correct_all = df.correct1 .& df.correct2

    df.subtree_relation = map(subtree_relation_code, df.path)

    df.diff1 = map(diff1, df.path)
    df.diff2 = map(diff2, df.value2, df.choice1)
    
    return df
end

function per_subject_then_group_mean(df::DataFrame; by_cols::Vector{Symbol}, y_col::Symbol)
    
    per_subj = combine(groupby(df, [:wid, by_cols...], sort=false), y_col => mean => :per_subject_mean)
    grp      = combine(groupby(per_subj, by_cols, sort=false), :per_subject_mean => mean => :group_mean)
    
    return grp
end

function summary_stats(df::DataFrame; scale_C=10_000.0, scale_D=1_000.0, scale_E=10_000.0)

    A_df = per_subject_then_group_mean(df; by_cols=[:diff1], y_col=:correct1)
    B_df = per_subject_then_group_mean(df; by_cols=[:diff2], y_col=:correct2)

    C_df = per_subject_then_group_mean(df[df.correct1 .== true, :]; by_cols=[:diff1], y_col=:rt1)
    C_df.group_mean .= C_df.group_mean ./ scale_C

    D_df = per_subject_then_group_mean(df[df.correct2 .== true, :]; by_cols=[:diff2], y_col=:rt2)
    D_df.group_mean .= D_df.group_mean ./ scale_D
    
    E_mask = (df.correct1 .== true) .& in.(df.subtree_relation, Ref([1,2,3]))
    E_df = per_subject_then_group_mean(df[E_mask, :]; by_cols=[:subtree_relation], y_col=:rt1)
    E_df.group_mean .= E_df.group_mean ./ scale_E

    return Dict{Symbol,DataFrame}(
        :A_df => A_df,
        :B_df => B_df,
        :C_df => C_df,
        :D_df => D_df,
        :E_df => E_df,
    )
end

function _rss_between(h::DataFrame, m::DataFrame, by_cols::Vector{Symbol})
    joined = outerjoin(h, m; on=by_cols, makeunique=true, matchmissing=:equal)
    hmean = coalesce.(joined.group_mean, 0.0)
    mmean = coalesce.(joined.group_mean_1, 0.0)
    return sum((mmean .- hmean) .^ 2)
end

function compute_rss(Human_Dict::Dict{Symbol,DataFrame}, Model_Dict::Dict{Symbol,DataFrame})
    
    rss_A = _rss_between(Human_Dict[:A_df], Model_Dict[:A_df], [:diff1])
    rss_B = _rss_between(Human_Dict[:B_df], Model_Dict[:B_df], [:diff2])
    rss_C = _rss_between(Human_Dict[:C_df], Model_Dict[:C_df], [:diff1])
    rss_D = _rss_between(Human_Dict[:D_df], Model_Dict[:D_df], [:diff2])
    rss_E = _rss_between(Human_Dict[:E_df], Model_Dict[:E_df], [:subtree_relation])
    
    return rss_A + rss_B + rss_C + rss_D + rss_E
end

function simulate_trials(trials, θ, config)
    
    model = Model(config.model_function, θ)
    rows = Vector{NamedTuple}(undef, length(trials))
    
    @inbounds for i in eachindex(trials)
        tr = trials[i]
        sim = simulate(model, tr)
        rows[i] = (;
            wid = tr.wid,
            rt1 = sim.rt1,
            rt2 = sim.rt2,
            choice1 = sim.choice1,
            choice2 = sim.choice2,
            value1 = tr.rewards[1:2],
            value2 = tr.rewards[3:6],
            rewards = tr.rewards,
            path = tr.path
        )
    end
    return DataFrame(rows)
end

function write_to_json(df::DataFrame, json_out::String)
    open(json_out, "w") do io
        for row in eachrow(df)
            obj = Dict(
                "wid" => row.wid,
                "rt1" => row.rt1,
                "rt2" => row.rt2,
                "choice1" => row.choice1,
                "choice2" => row.choice2,
                "value1" => collect(row.value1),
                "value2" => collect(row.value2),
                "rewards" => collect(row.rewards),
                "path" => collect(row.path),
                "best_path_idx" => get(row, :best_path_idx, missing),
                "correct1" => get(row, :correct1, missing),
                "correct2" => get(row, :correct2, missing),
                "correct_all" => get(row, :correct_all, missing),
                "subtree_relation" => get(row, :subtree_relation, missing),
                "diff1" => get(row, :diff1, missing),
                "diff2" => get(row, :diff2, missing),
            )
            write(io, JSON.json(obj))
            write(io, '\n')
        end
    end
end

function trials_to_observed_df(trials)
    rows = Vector{NamedTuple}(undef, length(trials))
    @inbounds for i in eachindex(trials)
        tr = trials[i]
        rows[i] = (;
            wid     = tr.wid,
            rt1     = tr.rt1,
            rt2     = tr.rt2,
            choice1 = tr.choice1,
            choice2 = tr.choice2,
            value1  = tr.rewards[1:2],
            value2  = tr.rewards[3:6],
            rewards = tr.rewards,
            path    = tr.path,
        )
    end
    return DataFrame(rows)
end


@everywhere function fit_model(wid, trials, model_name::String; optimizer::Symbol=:de, NumFuncEvals::Int=5000)
    
    # Get model configuration
    config = get_model_config(model_name)
    
    # Get hard bounds
    box = config.hard_bounds
    param_names = collect(keys(box.dims))
    n = n_free(box)

    lbs = zeros(n)
    ubs = ones(n)

    # Get plausible bounds
    pbox = config.plausible_bounds
    lower_dict = Dict(name => pbox.dims[name][1] for name in param_names)
    upper_dict = Dict(name => pbox.dims[name][2] for name in param_names)

    plbs = apply(box, lower_dict)
    pubs = apply(box, upper_dict)

    # Get initial parameters
    x0 = apply(box, config.initial_params)
    
    println("Worker $(myid()): Starting DE optimization using $model_name")
    
    # Tracker
    eval_count = Ref(0)

    Human_df = trials_to_observed_df(trials)
    Human_df = add_info(Human_df)
    Human_Dict = summary_stats(Human_df)

    function objective_function(x_unit)
        try
            eval_count[] += 1
            if eval_count[] % 100 == 0
                println("Worker $(myid()): $model_name - Evaluation $(eval_count[])")
            end

            x = Float64.(x_unit)
            θ_dict = box(x)
            θ = [θ_dict[name] for name in param_names]

            Model_df = simulate_trials(trials, θ, config)
            Model_df = add_info(Model_df)
            Model_Dict = summary_stats(Model_df)

            return compute_rss(Human_Dict, Model_Dict)
        
        catch e
            @error "Worker $(myid()): $model_name Exception: $(e)"
            return 1e6
        end
    end
    
    try
        search_range = [(plbs[i], pubs[i]) for i in 1:n]
        result = bboptimize(objective_function;
            # Method = :de_rand_1_bin,
            LowerBounds = lbs,
            UpperBounds = ubs,
            SearchRange = search_range,
            NumDimensions = n,
            MaxFuncEvals = NumFuncEvals,
            TraceMode = :compact,
            # TraceInterval = 50,
            FitnessTolerance = 1e-2,
        )
        xopt_unit = Float64.(best_candidate(result))
        fopt = Float64(best_fitness(result))

        real_xopt     = box(xopt_unit)
        xopt = [real_xopt[name] for name in param_names]
        
        return wid, xopt, fopt, param_names
        
    catch e
        @error "Worker $(myid()): DE failed" exception=(e, catch_backtrace())
        return wid, x0, 1e6, param_names
    end
end


function main(; data_file::String = "Tree2/data/Tree2_v3.json", model_name::String = "model6",
                output_file::Union{String, Nothing} = nothing,
                optimizer::Symbol = :de,
                NumFuncEvals::Int = 5000)
    
    config = get_model_config(model_name)
    
    if output_file === nothing
        output_file = "Tree2/results/rss/$(model_name)_RSS_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    end

    output_dir = dirname(output_file)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    trials = load_data(data_file)
    n_trials_total = length(trials)
    n_participants = length(unique([tr.wid for tr in trials]))
    
    wid, θ, rss, _ = fit_model("ALL", trials, model_name; optimizer = optimizer)

    # Simulate with best params and save JSON after add_info
    df_sim = simulate_trials(trials, θ, config)
    df_sim = add_info(df_sim)
    json_out = "Tree2/data/Tree2_sim/simulate_$(model_name)_RSS.json"
    write_to_json(df_sim, json_out)
    println("Simulated trials saved to: " * json_out)
    
    # Get parameter names from model configuration
    config = get_model_config(model_name)
    param_names = config.param_names
    n_params = length(param_names)
    param_count = config.param_nums
    
    column_names = [Symbol.(param_names); :rss; :param_count; :n_trials; :bic;]
    column_types = [fill(Float64, n_params); Float64; Int; Int; Float64]
    df = DataFrame([T[] for T in column_types], column_names)
    bic = param_count * log(n_participants) + n_participants * log(rss / n_participants)
    row_data = [θ; rss; param_count; n_trials_total; bic]
    push!(df, row_data)
    
    CSV.write(output_file, df)
    println("Results saved to: $output_file")
    
    println("\nFitting Summary:")
    println("Trials: $(n_trials_total)")
    println("RSS: $(round(rss, digits=2))")
    
    return df
end

# main(; data_file = "Tree2/data/Tree2_v3.json", model_name = "model11", optimizer = :de, NumFuncEvals = 10000) 