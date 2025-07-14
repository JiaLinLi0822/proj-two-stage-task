# fitting.jl
# Flexible model fitting script using the model configuration system
# Easy model switching and parameter management

using Distributed
using Dates

addprocs(4)

@everywhere begin
    include("ibs.jl")
    include("model.jl")
    include("likelihood.jl")
    include("data.jl")
    include("bads.jl")
    include("model_configs.jl")

    using JSON, DataFrames, CSV, Logging
    disable_logging(Logging.Warn)
end

# Fit a single subject using the specified model configuration.
# This function runs on all workers for parallel processing.
@everywhere function fit_subject(wid, trials, model_name::String)
    
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
            model = Model(config.model_function, θ)
    
            res = ibs_loglike(model, trials;
                              repeats  = 10,
                              max_iter = 1000,
                              ε        = 0.05,
                              rt_tol1  = 1000,
                              rt_tol2  = 1000,
                              min_multiplier = 0.8)
        
            neg_ll = res.neg_logp
    
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
        bads_result = optimize_bads(objective_function;
            x0 = x0,
            lower_bounds = lbs,
            upper_bounds = ubs, 
            plausible_lower_bounds = plbs,
            plausible_upper_bounds = pubs,
            max_fun_evals = 100,
            uncertainty_handling = true
        )
        
        result_dict = get_result(bads_result)
        xopt_unit_any = result_dict["x"]
        fopt = result_dict["fval"]
        
        xopt_unit     = Float64.(xopt_unit_any)
        real_xopt     = box(xopt_unit)

        xopt = [real_xopt[name] for name in param_names]

        println("Worker $(myid()): BADS completed for subject $wid ($(eval_count[]) evaluations)")
        println("Worker $(myid()): Subject $wid - Final θ = $xopt, negLL = $fopt")
        
        return wid, xopt, fopt, param_names
        
    catch e
        @error "Worker $(myid()): BADS failed for subject $wid" exception=(e, catch_backtrace())
        # If BADS fails, return the initial point and a large function value
        return wid, x0, 1e6, param_names
    end
end

"""
Run model fitting for all subjects using the specified model.
"""
function run_model_fitting(model_name::String; 
                          data_file::String = "data/Tree2_v3.json",
                          output_file::Union{String, Nothing} = nothing)
    
    # Validate model name
    config = get_model_config(model_name)
    
    # Set default output file name
    if output_file === nothing
        output_file = "results/results_$(model_name)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    end
    
    println("="^60)
    println("Model Fitting Configuration")
    println("="^60)
    println("Model: $model_name")
    println("Description: $(config.description)")
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
    
    # Run parallel fitting
    println("Starting parallel fitting...")
    pairs = collect(subject_trials)
    results = pmap(x -> fit_subject(x[1], x[2], model_name), pairs)
    
    # Collect and save results
    println("Collecting results...")
    
    # Get parameter names from model_configs
    config = get_model_config(model_name)
    param_names = config.param_names  # Use the dedicated param_names from model_configs
    n_params = length(param_names)
    
    # Create DataFrame with dynamic columns using real parameter names from model_configs
    column_names = [:wid; Symbol.(param_names); :neglogl]
    column_types = [String; fill(Float64, n_params); Float64]
    df = DataFrame([T[] for T in column_types], column_names)
    
    for (wid, θ, negll, _) in results
        row_data = [wid; θ; negll]
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
Compare multiple models by fitting them to the same data.
"""
function compare_models(model_names::Vector{String}; 
                       data_file::String = "data/Tree2_v3.json")
    
    println("="^60)
    println("Model Comparison")
    println("="^60)
    println("Models to compare: $(join(model_names, ", "))")
    println("Data file: $data_file")
    println("="^60)
    
    results = Dict{String, DataFrame}()
    
    for model_name in model_names
        println("\nFitting model: $model_name")
        output_file = "results/comparison_$(model_name)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
        results[model_name] = run_model_fitting(model_name; 
                                               data_file=data_file, 
                                               output_file=output_file)
    end
    
    # Create comparison summary
    println("\n" * "="^60)
    println("Model Comparison Summary")
    println("="^60)
    
    comparison_df = DataFrame(
        model = String[],
        mean_neglogl = Float64[],
        std_neglogl = Float64[],
        n_params = Int[],
        description = String[]
    )
    
    for model_name in model_names
        df = results[model_name]
        config = get_model_config(model_name)
        
        push!(comparison_df, (
            model_name,
            mean(df.neglogl),
            std(df.neglogl),
            length(config.parameter_box.dims),
            config.description
        ))
    end
    
    # Sort by mean negative log-likelihood (lower is better)
    sort!(comparison_df, :mean_neglogl)
    
    println(comparison_df)
    
    # Save comparison summary
    comparison_file = "results/model_comparison_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    CSV.write(comparison_file, comparison_df)
    println("\nComparison summary saved to: $comparison_file")
    
    return results, comparison_df
end

# Example usage functions
"""
Show available models and their descriptions.
"""
function show_models()
    list_models()
end

"""
Example: Fit a single model
"""
function example_single_model()
    results = run_model_fitting("model2")
end

"""
Example: Compare multiple models
"""
function example_model_comparison()
    # Compare different two-stage models
    model_names = ["model1", "model2", "model5", "model11"]
    results, comparison = compare_models(model_names)
    return results, comparison
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

example_single_model()