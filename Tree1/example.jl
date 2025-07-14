# example.jl
# Examples of how to use the flexible model fitting system

include("fitting.jl")

println("="^60)
println("Example Usage of Flexible Model Fitting System")
println("="^60)

# Example 1: Show all available models
println("\n1. Showing all available models:")
show_models()

# Example 2: Fit a single specific model
println("\n2. Example: Fitting a single model (model1)")
println("Uncomment the following line to run:")
println("# results_model1 = run_model_fitting(\"model1\")")

# Example 3: Compare multiple models
println("\n3. Example: Comparing multiple models")
println("Uncomment the following lines to run:")
println("# model_names = [\"model1\", \"model2\", \"model5\"]")
println("# results, comparison = compare_models(model_names)")

# Example 4: Fit model with custom settings
println("\n4. Example: Fitting with custom data file and output")
println("Uncomment the following lines to run:")
println("# results = run_model_fitting(\"model11\";")
println("#                             data_file=\"data/Tree2_v3.json\",")
println("#                             output_file=\"custom_results.csv\")")

# Example 5: Compare all two-stage models
println("\n5. Example: Comparing all two-stage models")
println("Uncomment the following lines to run:")
println("# two_stage_models = [\"model1\", \"model2\", \"model3\", \"model4\", \"model5\", \"model6\", \"model14\"]")
println("# results, comparison = compare_models(two_stage_models)")

# Example 6: Compare vigor models
println("\n6. Example: Comparing vigor-based models")
println("Uncomment the following lines to run:")
println("# vigor_models = [\"model11\", \"model12\", \"model13\"]")
println("# results, comparison = compare_models(vigor_models)")

# Example 7: Compare backward search models
println("\n7. Example: Comparing backward search models")
println("Uncomment the following lines to run:")
println("# backward_models = [\"model7\", \"model8\", \"model9\", \"model10\"]")
println("# results, comparison = compare_models(backward_models)")

println("\n" * "="^60)
println("Quick Start Instructions:")
println("="^60)
println("1. To fit a single model:")
println("   julia> results = run_model_fitting(\"model1\")")
println()
println("2. To compare multiple models:")
println("   julia> results, comparison = compare_models([\"model1\", \"model2\", \"model11\"])")
println()
println("3. To see parameter bounds for a specific model:")
println("   julia> config = get_model_config(\"model1\")")
println("   julia> display(config.parameter_box)")
println()
println("4. To get initial parameter values:")
println("   julia> x0, param_names = get_initial_params(config.parameter_box)")
println("="^60)

# Utility function to run a quick test
function quick_test(model_name::String = "model1")
    """
    Run a quick test with a single subject to verify everything works.
    """
    println("Running quick test with $model_name...")
    
    # Load a small subset of data for testing
    subject_trials = load_data_by_subject("data/Tree2_v3.json")
    first_subject = first(collect(subject_trials))
    
    # Get model configuration
    config = get_model_config(model_name)
    x0, param_names = get_initial_params(config.parameter_box)
    
    # Test the model function
    println("Testing model function...")
    model = Model(config.model_function, x0)
    trial = first_subject[2][1]  # First trial of first subject
    
    result = model.simulate(model.θ, trial.rewards)
    println("Model simulation result: $result")
    
    println("Quick test completed successfully!")
    return result
end

println("\nTo run a quick test, use: quick_test(\"model1\")")
println("="^60) 

quick_test("model2")