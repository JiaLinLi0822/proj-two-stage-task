using DataStructures: OrderedDict
include("box.jl")
include("model.jl")

"""
Model configuration system for easy model switching and parameter management.
Each model configuration includes:
- model_function: The actual model function to use
- hard_bounds: Box defining parameter hard lower and upper bounds and constraints
- plausible_bounds: Box with plausible lower and upper bounds for each parameter
- initial_params: Dict with initial parameter values
- description: Model names and descriptions
"""

struct ModelConfig
    model_function::Function
    hard_bounds::Box
    plausible_bounds::Box
    initial_params::Dict{Symbol, Float64}
    param_names::Vector{String}  # Parameter names for CSV storage
    param_nums::Int              # Number of parameters for BIC calculation
    description::String
end

# Helper function to create log-scale parameter bounds
log_bounds(low, high) = (low, high, :log)
linear_bounds(low, high) = (low, high)

# Define model configurations
const MODEL_CONFIGS = OrderedDict{String, ModelConfig}(

    # Two-stage models
    "model1" => ModelConfig(
        model1,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d1", "d2", "θ1", "θ2", "T1", "T2"],
        6,
        "Primary two-stage independent paths model"
    ),

    "model2" => ModelConfig(
        model2,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d1", "d2", "θ1", "θ2", "T1", "T2"],
        6,
        "Two-stage correlated paths model"
    ),

    "model3" => ModelConfig(
        model3,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0),
            :θ_prun => linear_bounds(1e-3, 2.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0),
            :θ_prun => linear_bounds(0.01, 1.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0, :θ_prun => 0.3),
        ["d1", "d2", "θ1", "θ2", "T1", "T2", "θ_prun"],
        7,
        "Two-stage independent paths with pruning"
    ),

    "model4" => ModelConfig(
        model4,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0),
            :θ_prun => linear_bounds(1e-3, 2.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0),
            :θ_prun => linear_bounds(0.01, 1.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0, :θ_prun => 0.3),
        ["d1", "d2", "θ1", "θ2", "T1", "T2", "θ_prun"],
        7,
        "Two-stage correlated paths with pruning"
    ),

    "model5" => ModelConfig(
        model5,
        Box(:d => log_bounds(1e-10, 1e-3),
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d => log_bounds(1e-8, 1e-4),
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d => 7e-5, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d", "θ1", "θ2", "T1", "T2"],
        5,
        "Two-stage independent paths with single drift rate"
    ),

    "model6" => ModelConfig(
        model6,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d1", "d2", "θ1", "θ2", "T1", "T2"],
        6,
        "Forward greedy search model"
    ),

    "model7" => ModelConfig(
        model7,
        Box(:d0 => log_bounds(1e-10, 1e-3),
            :d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ0 => linear_bounds(1e-3, 2.0),
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d0 => log_bounds(1e-8, 1e-4),
            :d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ0 => linear_bounds(0.01, 1.0),
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d0 => 7e-5, :d1 => 8e-5, :d2 => 6e-5, :θ0 => 0.4, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d0", "d1", "d2", "θ0", "θ1", "θ2", "T1", "T2"],
        8,
        "Backward search model"
    ),

    "model8" => ModelConfig(
        model8,
        Box(:d0 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ0 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d0 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ0 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d0 => 7e-5, :d2 => 6e-5, :θ0 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d0", "d2", "θ0", "θ2", "T1", "T2"],
        6,
        "Backward search with shared parameters"
    ),

    "model9" => ModelConfig(
        model9,
        Box(:d0 => log_bounds(1e-10, 1e-3),
            :d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ0 => linear_bounds(1e-3, 2.0),
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d0 => log_bounds(1e-8, 1e-4),
            :d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ0 => linear_bounds(0.01, 1.0),
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d0 => 7e-5, :d1 => 8e-5, :d2 => 6e-5, :θ0 => 0.4, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d0", "d1", "d2", "θ0", "θ1", "θ2", "T1", "T2"],
        8,
        "Backward search with reset"
    ),

    "model10" => ModelConfig(
        model10,
        Box(:d0 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ0 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d0 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ0 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d0 => 7e-5, :d2 => 6e-5, :θ0 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d0", "d2", "θ0", "θ2", "T1", "T2"],
        6,
        "Backward search with reset and shared parameters"
    ),

    "model11" => ModelConfig(
        model11,
        Box(:d => log_bounds(1e-10, 1e-3),
            :θ0 => linear_bounds(1e-3, 2.0),
            :vigor1 => linear_bounds(0.1, 5.0),
            :vigor2 => linear_bounds(0.1, 5.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d => log_bounds(1e-8, 1e-4),
            :θ0 => linear_bounds(0.01, 1.0),
            :vigor1 => linear_bounds(0.2, 3.0),
            :vigor2 => linear_bounds(0.2, 3.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d => 7e-5, :θ0 => 0.6, :vigor1 => 1.0, :vigor2 => 1.2, :T1 => 500.0, :T2 => 500.0),
        ["d", "θ0", "vigor1", "vigor2", "T1", "T2"],
        6,
        "One-stage parallel integration with vigor"
    ),

    "model12" => ModelConfig(
        model12,
        Box(:d => log_bounds(1e-10, 1e-3),
            :θ0 => linear_bounds(1e-3, 2.0),
            :vigor => linear_bounds(0.1, 5.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d => log_bounds(1e-8, 1e-4),
            :θ0 => linear_bounds(0.01, 1.0),
            :vigor => linear_bounds(0.2, 3.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d => 7e-5, :θ0 => 0.6, :vigor => 1.0, :T1 => 500.0, :T2 => 500.0),
        ["d", "θ0", "vigor", "T1", "T2"],
        5,
        "One-stage parallel integration with single vigor"
    ),

    "model13" => ModelConfig(
        model13,
        Box(:d => log_bounds(1e-10, 1e-3),
            :θ0 => linear_bounds(1e-3, 2.0),
            :vigor1 => linear_bounds(0.1, 5.0),
            :vigor2 => linear_bounds(0.1, 5.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0),
            :rating_sd => log_bounds(1e-3, 1.0)),
        Box(:d => log_bounds(1e-8, 1e-4),
            :θ0 => linear_bounds(0.01, 1.0),
            :vigor1 => linear_bounds(0.2, 3.0),
            :vigor2 => linear_bounds(0.2, 3.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0),
            :rating_sd => log_bounds(0.01, 0.5)),
        Dict(:d => 7e-5, :θ0 => 0.6, :vigor1 => 1.0, :vigor2 => 1.2, :T1 => 500.0, :T2 => 500.0, :rating_sd => 0.1),
        ["d", "θ0", "vigor1", "vigor2", "T1", "T2", "rating_sd"],
        7,
        "One-stage parallel integration with vigor and rating noise"
    ),

    "model14" => ModelConfig(
        model14,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :θ2 => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :θ2 => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :θ2 => 0.8, :T1 => 500.0, :T2 => 500.0),
        ["d1", "d2", "θ1", "θ2", "T1", "T2"],
        6,
        "Two-stage with average-based first stage"
    )
)

"""
Get model configuration by name.
"""
function get_model_config(model_name::String)
    if !haskey(MODEL_CONFIGS, model_name)
        available_models = join(keys(MODEL_CONFIGS), ", ")
        error("Model '$model_name' not found. Available models: $available_models")
    end
    return MODEL_CONFIGS[model_name]
end

"""
List all available models with descriptions.
"""
function list_models()
    println("Available models:")
    for (name, config) in MODEL_CONFIGS
        println("  $name: $(config.description)")
    end
end

"""
Convert parameter box to BADS-compatible bounds using Box for plausible bounds.
Returns lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds
"""
function get_bads_bounds(config::ModelConfig)
    param_names = collect(keys(config.hard_bounds.dims))
    n_params = length(param_names)
    
    lb = Float64[]
    ub = Float64[]
    plb = Float64[]
    pub = Float64[]
    
    for name in param_names
        # Get hard bounds from parameter box
        bounds = config.hard_bounds.dims[name]
        low, high = bounds[1], bounds[2]
        push!(lb, low)
        push!(ub, high)
        
        # Get plausible bounds from plausible_bounds Box
        plausible_bounds = config.plausible_bounds.dims[name]
        plausible_low, plausible_high = plausible_bounds[1], plausible_bounds[2]
        push!(plb, plausible_low)
        push!(pub, plausible_high)
    end
    
    return lb, ub, plb, pub, param_names
end

"""
Get BADS bounds for a box (backwards compatibility).
"""
function get_bads_bounds(box::Box)
    param_names = collect(keys(box.dims))
    n_params = length(param_names)
    
    lb = Float64[]
    ub = Float64[]
    plb = Float64[]
    pub = Float64[]
    
    for (name, bounds) in box.dims
        low, high = bounds[1], bounds[2]
        push!(lb, low)
        push!(ub, high)
        
        # Set plausible bounds as 10-90% of the range (in log space if log scale)
        if length(bounds) > 2 && bounds[3] == :log
            log_low, log_high = log(low), log(high)
            log_range = log_high - log_low
            push!(plb, exp(log_low + 0.1 * log_range))
            push!(pub, exp(log_low + 0.9 * log_range))
        else
            range_val = high - low
            push!(plb, low + 0.1 * range_val)
            push!(pub, low + 0.9 * range_val)
        end
    end
    
    return lb, ub, plb, pub, param_names
end

"""
Generate initial parameter values using directly specified values.
"""
function get_initial_params(config::ModelConfig)
    param_names = collect(keys(config.hard_bounds.dims))
    x0 = Float64[]
    
    for name in param_names
        initial_value = config.initial_params[name]
        push!(x0, initial_value)
    end
    
    return x0, param_names
end

"""
Generate initial parameter values for a box (backwards compatibility).
"""
function get_initial_params(box::Box)
    param_names = collect(keys(box.dims))
    x0 = Float64[]
    
    for (name, bounds) in box.dims
        low, high = bounds[1], bounds[2]
        
        # Generate initial value at midpoint (in log space if log scale)
        if length(bounds) > 2 && bounds[3] == :log
            log_mid = (log(low) + log(high)) / 2
            push!(x0, exp(log_mid))
        else
            mid = (low + high) / 2
            push!(x0, mid)
        end
    end
    
    return x0, param_names
end

"""
Convert parameter vector to named tuple based on model configuration.
"""
function params_to_named_tuple(params::Vector{Float64}, model_name::String)
    config = get_model_config(model_name)
    param_names = collect(keys(config.hard_bounds.dims))
    return NamedTuple{Tuple(param_names)}(params)
end

"""
Convert named tuple to parameter vector based on model configuration.
"""
function named_tuple_to_params(params::NamedTuple, model_name::String)
    config = get_model_config(model_name)
    param_names = collect(keys(config.hard_bounds.dims))
    return [getfield(params, name) for name in param_names]
end 