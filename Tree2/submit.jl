# submit.jl
using Distributed
using Dates

include("fitting.jl")

# Model and data configuration
models_to_fit = ["model10"]
data_file = "Tree2/data/Tree2_v3.json"

# Likelihood method: "ibs" or "pda"
likelihood_method = "pda"

# PDA-specific hyperparameters (only used when likelihood_method = "pda")
J = 1000                   # Number of simulations per trial for PDA
kde_model = :gaussian      # KDE model: :product (Epanechnikov) or :gaussian (multivariate Gaussian)
bw_rule = :silverman       # Bandwidth rule for Gaussian KDE: :scott or :silverman
eps_floor = 1e-16          # Floor value for numerical stability
lambda = 1.0               # Lambda for PDA
optimizer = :BADS          # Optimizer: :BADS or :DE

for model_name in models_to_fit
    println("Fitting $model_name...")
    t0 = now()
    try
        if likelihood_method == "pda"
            result_df = run_model_fitting(model_name;
                data_file=data_file,
                likelihood_method=likelihood_method,
                J=J,
                kde_mode=kde_model,
                bw_rule=bw_rule,
                eps_floor=eps_floor,
                lambda=lambda,
                optimizer=optimizer)
        else
            result_df = run_model_fitting(model_name;
                data_file=data_file,
                likelihood_method=likelihood_method,
                optimizer=optimizer)
        end
        dt_min = (now() - t0).value / (1000 * 60)
        println("$model_name completed ($(round(dt_min, digits=1)) min)")
    catch e
        println("$model_name failed: $e")
    end
end