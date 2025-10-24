#!/usr/bin/env julia
using Distributed
using Logging
using Statistics

# addprocs(8)

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
    # include("bads.jl")
    include("model_configs.jl")
    
    Logging.disable_logging(Logging.Warn)
end

@everywhere function optimize_bads(f::Function;
    x0::AbstractVector,
    lower_bounds::AbstractVector,
    upper_bounds::AbstractVector,
    plausible_lower_bounds::AbstractVector = lower_bounds,
    plausible_upper_bounds::AbstractVector = upper_bounds,
    max_fun_evals::Integer = 1000,
    uncertainty_handling::Bool = false
)
    pybads = pyimport("pybads")

    py"""
    import numpy as _np

    _jl_callback = None

    def __set_jl_callback__(cb):
        global _jl_callback
        _jl_callback = cb

    def __py_obj__(x):
        x = _np.asarray(x, dtype=float).ravel().tolist()
        return float(_jl_callback(x))
    """

    pymain = pyimport("__main__")
    pymain.__set_jl_callback__(pyfunction(f, Vector{Float64}))

    BADS = pybads.BADS
    b = BADS(
        pymain.__py_obj__, collect(x0);
        lower_bounds           = collect(lower_bounds),
        upper_bounds           = collect(upper_bounds),
        plausible_lower_bounds = collect(plausible_lower_bounds),
        plausible_upper_bounds = collect(plausible_upper_bounds),
        options = Dict(
            "max_fun_evals"        => Int(max_fun_evals),
            "uncertainty_handling" => uncertainty_handling
        )
    )

    res = b.optimize()
    out = Dict{String,Any}()
    out["x"]         = Vector{Float64}(Array(res["x"]))
    out["fval"]      = Float64(res["fval"])
    out["exit_flag"] = Int(get(res, "exit_flag", 0))
    out["niters"]    = Int(get(res, "niters", 0))
    out["nevals"]    = Int(get(res, "nevals", 0))
    return out
end

# ----------------- First Passage Time Density -----------------
@everywhere function fpt_density_at(t::Float64; μ::Float64, θ::Float64, upper::Bool, σ::Float64=0.01)
    if t <= 0.0
        return 0.0
    end

    μs = μ/σ
    θs = θ/σ

    dt  = 1
    d   = ConstDrift(μs, dt)
    Bs  = ConstSymBounds(θs, dt)
    
    if upper
        return pdfu(d, Bs, t)
    else
        return pdfl(d, Bs, t)
    end
end

# ----------------- Single trial loglik -----------------
@everywhere function loglik_trial_stagewise(tr::Trial, φ::Vector{Float64}; eps::Float64=1e-16)
    
    σ = 0.01
    σeff = sqrt(2)*σ
    d1, d2, θ1, θ2, T1, T2  = φ

    r1,r2,r3,r4,r5,r6 = tr.rewards

    # T1 = round(Int, T1)
    # T2 = round(Int, T2)

    # ---- stage1 ----
    μ1   = d1 * (r1 - r2)
    t1 = Float64(tr.rt1) - Float64(T1)

    upper1 = (tr.choice1 == 1)
    g1 = fpt_density_at(t1; μ=μ1, θ=θ1, upper=upper1, σ=σeff)
    
    max_g1 = max(g1, eps)

    # ---- stage2 ----
    t2 = Float64(tr.rt2) - Float64(T2)

    if tr.choice1 == 1
        # (LL,LR)
        μ2 = d2 * (r3 - r4)
        upper2 = (tr.choice2 == 1)
    else
        # (RL,RR)
        μ2 = d2 * (r5 - r6)
        upper2 = (tr.choice2 == 3)
    end
    g2 = fpt_density_at(t2; μ=μ2, θ=θ2, upper=upper2, σ=σeff)
    
    max_g2 = max(g2, eps)

    # println("g1: $g1, g2: $g2")

    return log(max_g1) + log(max_g2)
end

# ----------------- Total loglik -----------------
@everywhere total_loglik(trials::Vector{Trial}, φ::Vector{Float64}) =
    sum(loglik_trial_stagewise(tr, φ) for tr in trials)

# ----------------- Unified optimization (BADS or DE) -----------------
@everywhere function fit_with_bads(wid::String, trials::Vector{Trial}; optimizer::Symbol = :bads, log_to_file::Bool = true)
    # Get model6 configuration
    config = get_model_config("model6")
    
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
    
    method_label = optimizer == :de ? "DE" : "BADS"
    if !(optimizer in (:bads, :de))
        error("Unknown optimizer: $(optimizer). Use :bads or :de")
    end
    println("Worker $(myid()): Starting $(method_label) optimization for subject $wid using model6")
    
    # Tracker
    eval_count = Ref(0)
    
    # Build objective with optional file logging
    log_file = method_label == "DE" ?
        "results/Tree2/log_DE_$(wid)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv" :
        "results/Tree2/log_BADS_$(wid)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    
    if log_to_file
        log_dir = dirname(log_file)
        if !isdir(log_dir)
            mkpath(log_dir)
        end
    end

    # Objective factory that writes to io only if log_to_file is true
    function make_objective(io)
        function objective_function(x_unit)
            try
                eval_count[] += 1
                if eval_count[] % 100 == 0
                    println("Worker $(myid()): Subject $wid ($(method_label)) - Evaluation $(eval_count[])")
                end
                x = Float64.(x_unit)
                θ_dict = box(x)
                θ = [θ_dict[name] for name in param_names]
                neg_ll = -total_loglik(trials, θ)
                if !isfinite(neg_ll)
                    @error "Worker $(myid()): Bad negative log-likelihood estimate for $wid: $neg_ll"
                    neg_ll = 1e6
                end
                if log_to_file
                    println(io, "$(eval_count[]),$neg_ll,$(join(θ, ","))")
                    flush(io)
                end
                return Float64(neg_ll)
            catch e
                @error "Worker $(myid()): ($(method_label)) Exception for $wid: $(e)"
                if log_to_file
                    zero_params = join(zeros(length(param_names)), ",")
                    println(io, "$(eval_count[]),1e6,$zero_params")
                    flush(io)
                end
                return 1e6
            end
        end
        return objective_function
    end

    # Shared optimization runner using the provided IO
    function run_opt(io)
        objective_function = make_objective(io)
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
                MaxFuncEvals = 1000,
                TraceMode = :silent
            )
            xopt_unit = Float64.(best_candidate(result))
            fopt = Float64(best_fitness(result))
        end
        
        real_xopt = box(xopt_unit)
        xopt = [real_xopt[name] for name in param_names]
        
        println("Worker $(myid()): $(method_label) completed for subject $wid ($(eval_count[]) evaluations)")
        println("Worker $(myid()): Subject $wid ($(method_label)) - Final θ = $xopt, negLL = $fopt")
        
        if log_to_file
            println("Worker $(myid()): ($(method_label)) Optimization log saved to: $log_file")
        end
        return wid, xopt, fopt, param_names, method_label
    end

    try
        if log_to_file
            open(log_file, "w") do io
                param_header = join(param_names, ",")
                println(io, "evaluation,neg_loglik,$param_header")
                return run_opt(io)
            end
        else
            return run_opt(devnull)
        end
    catch e
        @error "Worker $(myid()): $(method_label) failed for subject $wid" exception=(e, catch_backtrace())
        return wid, x0, 1e6, param_names, method_label
    end
end

# ----------------- Main fitting function -----------------
function main(; data_file::String = "data/Tree2_v3.json",
                output_file::Union{String, Nothing} = nothing,
                optimizer::Symbol = :bads,
                log_to_file::Bool = true)
    
    config = get_model_config("model6")
    
    if output_file === nothing
        output_file = "results/Tree2/model6_test_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    end

    output_dir = dirname(output_file)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    subject_trials = load_data_by_subject(data_file)
    trial_counts = count_trials_per_participant(data_file)
    
    pairs = collect(subject_trials)

    if optimizer == :bads
        results = pmap(x -> fit_with_bads(x[1], x[2]; log_to_file = log_to_file), pairs)
    elseif optimizer == :de
        results = pmap(x -> fit_with_bads(x[1], x[2]; optimizer = :de, log_to_file = log_to_file), pairs)
    else
        error("Unknown optimizer: $(optimizer). Use :bads or :de")
    end
    
    # Get parameter names from model6 configuration
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
    
    println("\nFitting Summary:")
    total_results = length(results)
    println("Successfully fitted $(total_results) subjects")
    println("Mean negative log-likelihood: $(round(mean(df.neglogl), digits=2))")
    println("Std negative log-likelihood: $(round(std(df.neglogl), digits=2))")
    
    return df
end

# Run the main fitting
results = main(;data_file = "data/Tree2_v3.json", log_to_file = false, optimizer = :bads)

