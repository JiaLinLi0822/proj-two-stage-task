#!/usr/bin/env julia
using Distributed
using Logging
using Statistics

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
    using PyCall

    LinearAlgebra.BLAS.set_num_threads(1)
    
    include("data.jl")
    include("fpt.jl")
    include("model6_config.jl")
    include("model6.jl")
    include("bads.jl")

    Logging.disable_logging(Logging.Warn)
end

@everywhere total_loglik(trials::Vector{Trial}, φ::Vector{Float64}) =
    sum(loglik_trial_stagewise(tr, φ) for tr in trials)

@everywhere function fitting(wid::String, trials::Vector{Trial}; optimizer::Symbol = :bads)

    config = get_model_config("model6")
    
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
    
    method_label = optimizer == :de ? "DE" : "BADS"
    if !(optimizer in (:bads, :de))
        error("Unknown optimizer: $(optimizer). Use :bads or :de")
    end
    println("Worker $(myid()): Starting $(method_label) optimization for subject $wid using model6")
    
    objective_function = function(x_unit::Vector{Float64})
        real_params = box(x_unit)
        φ = [real_params[name] for name in param_names]
        return -total_loglik(trials, φ)
    end
    
    if optimizer == :bads
            result_dict = optimize_bads(objective_function;
                x0 = x0,
                lower_bounds = lbs,
                upper_bounds = ubs, 
                plausible_lower_bounds = plbs,
                plausible_upper_bounds = pubs,
                max_fun_evals = 1000,
                uncertainty_handling = false
            )
            xopt_unit_any = result_dict["x"]
            fopt = result_dict["fval"]
            xopt_unit = Float64.(xopt_unit_any)

        
    elseif optimizer == :de
            search_range = [(plbs[i], pubs[i]) for i in 1:n]
            result = bboptimize(objective_function;
                Method = :de_rand_1_bin,
                LowerBounds = lbs,
                UpperBounds = ubs,
                SearchRange = search_range,
                NumDimensions = n,
                MaxFuncEvals = 10000,
                TraceMode = :silent
            )
            xopt_unit = Float64.(best_candidate(result))
            fopt = Float64(best_fitness(result))
    end

    real_xopt = box(xopt_unit)
    xopt = [real_xopt[name] for name in param_names]
            
    return wid, xopt, fopt, param_names, method_label
end

function main(; data_file::String = "data/Tree2_v3.json",
                output_file::Union{String, Nothing} = nothing,
                optimizer::Symbol = :bads)
    
    config = get_model_config("model6")

    if output_file === nothing
        output_file = "pda/results/model6_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    end

    output_dir = dirname(output_file)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    subject_trials = load_data_by_subject(data_file)
    trial_counts = count_trials_per_participant(data_file)
    
    pairs = collect(subject_trials)

    if optimizer == :bads
        results = pmap(x -> fitting(x[1], x[2]; optimizer = :bads), pairs)
    elseif optimizer == :de
        results = pmap(x -> fitting(x[1], x[2]; optimizer = :de), pairs)
    else
        error("Unknown optimizer: $(optimizer). Use :bads or :de")
    end
    
    config = get_model_config("model6")
    param_names = config.param_names
    n_params = length(param_names)
    param_count = config.param_nums
    
    column_names = [:wid; Symbol.(param_names); :neglogl; :param_count; :n_trials; :bic; :method]
    column_types = [String; fill(Float64, n_params); Float64; Int; Int; Float64; String]
    df = DataFrame([T[] for T in column_types], column_names)
    
    for (wid, θ, negll, _, method) in results
        n_trials = get(trial_counts, wid, 0)
        bic = param_count * log(n_trials) + 2 * negll
        row_data = [wid; θ; negll; param_count; n_trials; bic; method]
        push!(df, row_data)
    end
    
    CSV.write(output_file, df)
    println("Results saved to: $output_file")
    
    return df
end

# Run the main fitting
results = main(;data_file = "pda/data/Tree2_v3.json", optimizer = :de)

