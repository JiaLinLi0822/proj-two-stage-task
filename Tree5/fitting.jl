# fitting.jl

using Distributed
using Dates

@everywhere begin
    include("ibs.jl")
    include("model.jl")
    include("likelihood.jl")
    include("data.jl")
    include("bads.jl")
    include("model_configs.jl")
    include("pda.jl")

    using JSON, DataFrames, CSV, Logging, Random, BlackBoxOptim
    disable_logging(Logging.Warn)
end


@everywhere function fit_subject(wid, trials, model_name::String, likelihood_method::String="ibs";
                                kde_mode::Symbol=:gaussian, bw_rule::Symbol=:silverman, J::Int=500, eps_floor::Float64=1e-16, lambda::Float64=1.0, optimizer::Symbol=:DE)
    
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
    
    println("Worker $(myid()): Starting BADS optimization for subject $wid using $model_name")
    
    # Tracker
    eval_count = Ref(0)
    
    # Objective function
    function objective_function(x_unit)
        try
            eval_count[] += 1
            if eval_count[] % 100 == 0
                println("Worker $(myid()): Subject $wid - Evaluation $(eval_count[])")
            end
        
            # Create model using the configured model function
            x = Float64.(x_unit)
            θ_dict = box(x)
            θ = [θ_dict[name] for name in param_names]
            
            if likelihood_method == "ibs"
                model = Model(config.model_function, θ)
        
                res = ibs_loglike(model, trials;
                                  repeats  = 10,
                                  max_iter = 1000,
                                  ε        = 0.05,
                                  rt_tol1  = 1000,
                                  rt_tol2  = 1000,
                                  min_multiplier = 0.8)
            
                neg_ll = res.neg_logp
            elseif likelihood_method == "pda"

                ll =  pda_loglike(θ, trials, 
                                  config.model_function;
                                  J=J, 
                                  kde_mode=kde_mode, 
                                  bw_rule=bw_rule, 
                                  eps_floor=eps_floor,
                                  lambda=lambda)
                neg_ll = -ll
            else
                error("Unknown likelihood method: $likelihood_method. Use 'ibs' or 'pda'.")
            end
            
            @info "DEBUG_RETURN_NLL" wid=wid eval=eval_count[] nll=neg_ll method=likelihood_method
            if !isfinite(neg_ll) || neg_ll < 0
                @error "Worker $(myid()): Bad negative log-likelihood estimate for $wid: $neg_ll"
                return 1e6
            end
    
            return Float64(neg_ll)
    
        catch e
            @error "Worker $(myid()): Exception for $wid: $(e)"
            return 1e6
        end
    end
    
    try
        optimizer_label = optimizer == :DE ? "DE" : "BADS"
        if optimizer == :BADS
            bads_result = optimize_bads(objective_function;
                x0 = x0,
                lower_bounds = lbs,
                upper_bounds = ubs, 
                plausible_lower_bounds = plbs,
                plausible_upper_bounds = pubs,
                max_fun_evals = 1000,
                uncertainty_handling = false,
                specify_target_noise = false,
            )
            result_dict = get_result(bads_result)
            xopt_unit_any = result_dict["x"]
            fopt = result_dict["fval"]
            xopt_unit = Float64.(xopt_unit_any)

        elseif optimizer == :DE
            search_range = [(plbs[i], pubs[i]) for i in 1:n]
            result = bboptimize(objective_function;
                Method = :de_rand_1_bin,
                LowerBounds = lbs,
                UpperBounds = ubs,
                SearchRange = search_range,
                NumDimensions = n,
                MaxFuncEvals = 2000,
                TraceMode = :compact,
                # TraceInterval = 50,
                FitnessTolerance = 1e-2,
            )
            xopt_unit = Float64.(best_candidate(result))
            fopt = Float64(best_fitness(result))
        end

        real_xopt     = box(xopt_unit)
        xopt = [real_xopt[name] for name in param_names]

        println("Worker $(myid()): $optimizer completed for subject $wid ($(eval_count[]) evaluations)")
        println("Worker $(myid()): Subject $wid - $optimizer_label - Final θ = $xopt, negLL = $fopt")
        
        return wid, xopt, fopt, param_names, optimizer_label
        
    catch e
        @error "Worker $(myid()): $optimizer_label failed for subject $wid" exception=(e, catch_backtrace())
        # If BADS fails, return the initial point and a large function value
        return wid, x0, 1e6, param_names, optimizer_label
    end
end

"""
Run model fitting for all subjects using the specified model.
"""
function run_model_fitting(model_name::String; 
                          data_file::String = "Tree1/data/Tree1_v3.json",
                          output_file::Union{String, Nothing} = nothing,
                          likelihood_method::String = "ibs",
                          kde_mode::Symbol = :gaussian,
                          bw_rule::Symbol = :silverman,
                          J::Int = 1000,
                          eps_floor::Float64 = 1e-16,
                          lambda::Float64 = 1.0,
                          optimizer::Symbol = :DE)
    
    # Validate model name
    config = get_model_config(model_name)
    
    # Set default output file name
    if output_file === nothing
        if likelihood_method == "pda"
            output_file = "Tree1/results/pda/$(model_name)_$(likelihood_method)_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
        else
            output_file = "Tree1/results/ibs/$(model_name)_$(likelihood_method)_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
        end
    end

    # Create output directory if it doesn't exist
    output_dir = dirname(output_file)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    println("="^60)
    println("Model Fitting Configuration")
    println("="^60)
    println("Model: $model_name")
    println("Description: $(config.description)")
    println("Likelihood method: $likelihood_method")
    println("Optimizer: $optimizer")
    if likelihood_method == "pda"
        println("PDA Configuration:")
        println("  KDE mode: $kde_mode")
        println("  Bandwidth rule: $bw_rule")
        println("  Eps floor: $eps_floor")
        println("  Simulations per trial: $J")
    end
    println("Data file: $data_file")
    println("Output file: $output_file")
    println("Parameter bounds:")
    
    # Display parameter configuration
    for (name, bounds) in config.hard_bounds.dims
        scale = length(bounds) > 2 && bounds[3] == :log ? " (log scale)" : ""
        println("  $name: [$(bounds[1]), $(bounds[2])]$scale")
    end
    println("="^60)
    
    # Load data
    println("Loading data...")
    subject_trials = load_data_by_subject(data_file)
    println("Loaded data for $(length(subject_trials)) subjects")
    
    # Count trials per participant for BIC calculation
    println("Counting trials per participant for BIC calculation...")
    trial_counts = count_trials_per_participant(data_file)
    
    # Run parallel fitting
    println("Starting parallel fitting...")
    pairs = collect(subject_trials)
    results = pmap(x -> fit_subject(x[1], x[2], model_name, likelihood_method; 
                                   kde_mode=kde_mode, bw_rule=bw_rule, J=J, eps_floor=eps_floor, lambda=lambda, optimizer=optimizer), pairs)
    
    # Collect and save results
    println("Collecting results...")
    
    # Get parameter names from model_configs
    config = get_model_config(model_name)
    param_names = config.param_names
    n_params = length(param_names)
    param_count = config.param_nums
    
    column_names = [:wid; Symbol.(param_names); :neglogl; :param_count; :n_trials; :bic; :optimizer]
    column_types = [String; fill(Float64, n_params); Float64; Int; Int; Float64; String]
    df = DataFrame([T[] for T in column_types], column_names)
    
    for (wid, θ, negll, _, optimizer) in results
        n_trials = get(trial_counts, wid, 0)
        bic = param_count * log(n_trials) + 2 * negll
        row_data = [wid; θ; negll; param_count; n_trials; bic; optimizer]
        push!(df, row_data)
    end
    
    # Save results
    CSV.write(output_file, df)
    println("Results saved to: $output_file")
    
    # Display summary statistics
    println("\nFitting Summary:")
    println("Successfully fitted $(length(results)) subjects")
    println("Mean negative log-likelihood: $(round(mean(df.neglogl), digits=2))")
    println("Std negative log-likelihood: $(round(std(df.neglogl), digits=2))")
    
    return df
end

"""
    run_pda_fitting(model_name; kwargs...)

Convenience function to run PDA-based model fitting.
"""
function run_pda_fitting(model_name::String; kde_mode::Symbol=:gaussian, bw_rule::Symbol=:silverman, J::Int=500, eps_floor::Float64=1e-16, lambda::Float64=1.0, optimizer::Symbol=:BADS, kwargs...)
    return run_model_fitting(model_name; likelihood_method="pda", kde_mode=kde_mode, bw_rule=bw_rule, J=J, eps_floor=eps_floor, lambda=lambda, optimizer=optimizer, kwargs...)
end

"""
    run_ibs_fitting(model_name; kwargs...)

Convenience function to run IBS-based model fitting (default method).
"""
function run_ibs_fitting(model_name::String; optimizer::Symbol=:BADS, kwargs...)
    return run_model_fitting(model_name; likelihood_method="ibs", optimizer=optimizer, kwargs...)
end


# Print usage instructions when script is loaded
println("="^60)
println("Flexible Model Fitting System")
println("="^60)
println("Available functions:")
println("  show_models()              - List all available models")
println("  run_model_fitting(model)   - Fit a specific model")
println("  compare_models([models])   - Compare multiple models")
println("  example_single_model()     - Example: fit model1")
println("  example_model_comparison() - Example: compare models")
println("="^60)
println("To get started, try: show_models()")
println("="^60)

# Example usage:
# run_pda_fitting("model6"; kde_mode=:gaussian, bw_rule=:silverman, J=1000, lambda=1.0, optimizer=:DE)
# run_ibs_fitting("model1"; optimizer=:DE)