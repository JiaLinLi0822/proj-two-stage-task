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

Note: These configurations work with both IBS and PDA likelihood methods.
For PDA, the model function is used within simulate_batch_model() to generate
synthetic data for the pseudo-likelihood approximation.
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

# Function to get model configurations
function get_model_configs()
    return OrderedDict{String, ModelConfig}(

    # Two-stage models
    "model1" => ModelConfig(
        model1,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :Δ => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :Δ => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :Δ => 0.3, :T1 => 500.0, :T2 => 500.0),
        ["d1", "d2", "θ1", "Δ", "T1", "T2"],
        6,
        "Primary two-stage independent paths model"
    ),

    "model2" => ModelConfig(
        model2,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :Δ => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :Δ => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :Δ => 0.3, :T1 => 500.0, :T2 => 500.0),
        ["d1", "d2", "θ1", "Δ", "T1", "T2"],
        6,
        "Two-stage correlated paths model"
    ),

    "model3" => ModelConfig(
        model3,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :Δ => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0),
            :θ_prun => linear_bounds(1e-3, 2.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :Δ => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0),
            :θ_prun => linear_bounds(0.01, 1.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :Δ => 0.3, :T1 => 500.0, :T2 => 500.0, :θ_prun => 0.3),
        ["d1", "d2", "θ1", "Δ", "T1", "T2", "θ_prun"],
        7,
        "Two-stage independent paths with pruning"
    ),

    "model4" => ModelConfig(
        model4,
        Box(:d1 => log_bounds(1e-10, 1e-3),
            :d2 => log_bounds(1e-10, 1e-3), 
            :θ1 => linear_bounds(1e-3, 2.0),
            :Δ => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0),
            :θ_prun => linear_bounds(1e-3, 2.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :Δ => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0),
            :θ_prun => linear_bounds(0.01, 1.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :Δ => 0.3, :T1 => 500.0, :T2 => 500.0, :θ_prun => 0.3),
        ["d1", "d2", "θ1", "Δ", "T1", "T2", "θ_prun"],
        7,
        "Two-stage correlated paths with pruning"
    ),

    "model5" => ModelConfig(
        model5,
        Box(:d => log_bounds(1e-10, 1e-3),
            :θ1 => linear_bounds(1e-3, 2.0),
            :Δ => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d => log_bounds(1e-8, 1e-4),
            :θ1 => linear_bounds(0.01, 1.0),
            :Δ => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d => 7e-5, :θ1 => 0.5, :Δ => 0.3, :T1 => 500.0, :T2 => 500.0),
        ["d", "θ1", "Δ", "T1", "T2"],
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
            :Δ => linear_bounds(1e-3, 2.0),
            :T1 => linear_bounds(10.0, 10000.0),
            :T2 => linear_bounds(10.0, 10000.0)),
        Box(:d1 => log_bounds(1e-8, 1e-4),
            :d2 => log_bounds(1e-8, 1e-4), 
            :θ1 => linear_bounds(0.01, 1.0),
            :Δ => linear_bounds(0.01, 1.0),
            :T1 => linear_bounds(50.0, 8000.0),
            :T2 => linear_bounds(50.0, 8000.0)),
        Dict(:d1 => 8e-5, :d2 => 6e-5, :θ1 => 0.5, :Δ => 0.3, :T1 => 500.0, :T2 => 500.0),
        ["d1", "d2", "θ1", "Δ", "T1", "T2"],
        6,
        "Two-stage with average-based first stage"
    )
)
end

"""
Get model configuration by name.
"""
function get_model_config(model_name::String)
    configs = get_model_configs()
    if !haskey(configs, model_name)
        available_models = join(keys(configs), ", ")
        error("Model '$model_name' not found. Available models: $available_models")
    end
    return configs[model_name]
end

"""
List all available models with descriptions.
"""
function list_models()
    configs = get_model_configs()
    println("Available models:")
    for (name, config) in configs
        println("  $name: $(config.description)")
    end
end

