using DataStructures: OrderedDict
include("box.jl")
include("model6.jl")

struct ModelConfig
    model_function::Function
    hard_bounds::Box
    plausible_bounds::Box
    initial_params::Dict{Symbol, Float64}
    param_names::Vector{String} 
    param_nums::Int             
    description::String
end

log_bounds(low, high) = (low, high, :log)
linear_bounds(low, high) = (low, high)

function get_model_configs()
    return OrderedDict{String, ModelConfig}(

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

)
end

