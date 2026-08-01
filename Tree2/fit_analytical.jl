#!/usr/bin/env julia
# Fit Model 6 using analytical likelihood and BADS optimizer

using Distributed
using Dates

include("fitting.jl")

# Model and data configuration
model_name = "model6"
data_file = "Tree2/data/Tree2_v3.json"
likelihood_method = "analytical"
optimizer = :BADS
output_file = nothing #"Tree2/results/analytical/model6_analytical_BADS.csv"

println("="^70)
println("Analytical Likelihood Fitting")
println("="^70)
println("Model: $model_name")
println("Likelihood method: $likelihood_method")
println("Optimizer: $optimizer")
println("Data file: $data_file")
println("="^70)
println()

# Run fitting
t0 = now()
try
    result_df = run_model_fitting(model_name;
        data_file=data_file,
        likelihood_method=likelihood_method,
        optimizer=optimizer,
        output_file=output_file)
    
    dt_min = (now() - t0).value / (1000 * 60)
    println("\n" * "="^70)
    println("Fitting completed successfully!")
    println("Total time: $(round(dt_min, digits=1)) minutes")
    println("Results saved to: $(result_df === nothing ? "N/A" : "see output above")")
    println("="^70)
catch e
    println("\n" * "="^70)
    println("Fitting failed: $e")
    println("="^70)
    rethrow(e)
end
