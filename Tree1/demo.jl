# demo.jl - IBS Log-Likelihood Estimation Demo
# This demo shows how to use Inverse Biased Sampling (IBS) to estimate log-likelihood

using Random
Random.seed!(42)  # For reproducible results

# Include the necessary modules
include("ibs.jl")
include("models.jl")

using .IBS
using .DecisionModel

println("=" ^ 60)
println("IBS Log-Likelihood Estimation Demo")
println("=" ^ 60)

# =============================================================================
# Step 1: Generate Synthetic Data
# =============================================================================
println("\n1. Generating synthetic data...")

# True model parameters [d1, d2, th1, th2, T1, T2]
true_params = [8e-6, 6e-6, 0.12, 0.18, 188.0, 236.0]
println("True parameters: $true_params")

# Create design matrix (reward structure for trials)
n_trials = 50
design_matrix = zeros(Float64, n_trials, 6)

# Generate random reward values for demonstration
Random.seed!(123)
for i in 1:n_trials
    # Random rewards for [R_L, R_R, R_LL, R_LR, R_RL, R_RR]
    design_matrix[i, 1] = rand() * 10      # R_L (left option value)
    design_matrix[i, 2] = rand() * 10      # R_R (right option value)
    design_matrix[i, 3] = rand() * 5       # R_LL (left-left outcome)
    design_matrix[i, 4] = rand() * 5       # R_LR (left-right outcome)
    design_matrix[i, 5] = rand() * 5       # R_RL (right-left outcome)
    design_matrix[i, 6] = rand() * 5       # R_RR (right-right outcome)
end

println("Design matrix shape: $(size(design_matrix))")
println("First 3 trials:")
for i in 1:3
    println("  Trial $i: $(round.(design_matrix[i, :], digits=2))")
end

# Generate observed data using the true model
println("\nGenerating observed responses using true parameters...")
observed_data = DecisionModel.model1(true_params, design_matrix)
println("Response matrix shape: $(size(observed_data))")
println("Response format: [choice1, rt1, choice2, rt2]")
println("First 3 responses:")
for i in 1:3
    println("  Trial $i: $(round.(observed_data[i, :], digits=2))")
end

# =============================================================================
# Step 2: Set up IBS Sampler
# =============================================================================
println("\n2. Setting up IBS sampler...")

# Create a wrapper model with lapses (more realistic)
rt1_min, rt1_max = extrema(observed_data[:, 2])
rt2_min, rt2_max = extrema(observed_data[:, 4])
println("RT1 range: $(round(rt1_min, digits=1)) - $(round(rt1_max, digits=1))")
println("RT2 range: $(round(rt2_min, digits=1)) - $(round(rt2_max, digits=1))")

# Create lapse model (adds noise to simulate lapses)
sample_with_lapse = DecisionModel.make_lapse_wrapper(
    DecisionModel.model1;
    epsilon=0.05,  # 5% lapse rate
    rt1_range=(rt1_min, rt1_max),
    rt2_range=(rt2_min, rt2_max)
)

# Set tolerance for RT matching
rt1_tol, rt2_tol = 200.0, 200.0

# Create IBS sampler
ibs_sampler = IBSSampler(
    sample_with_lapse,
    observed_data,
    design_matrix;
    max_iter=1000,        # Maximum iterations per repetition
    max_time=5.0,         # Maximum time in seconds
    vectorized=true,      # Use vectorized sampling
    rt1_tol=rt1_tol,
    rt2_tol=rt2_tol
)

println("IBS sampler created successfully!")

# =============================================================================
# Step 3: Estimate Log-Likelihood for True Parameters
# =============================================================================
println("\n3. Estimating log-likelihood for true parameters...")

# Basic estimation (returns negative log-likelihood)
neg_logl_true = ibs_sampler(true_params; num_reps=10)
println("Negative log-likelihood (true params): $(round(neg_logl_true, digits=4))")

# Estimation with variance
neg_logl_true_var, variance = ibs_sampler(true_params; num_reps=10, additional_output="var")
println("With variance - Neg-LL: $(round(neg_logl_true_var, digits=4)), Var: $(round(variance, digits=6))")

# Estimation with standard deviation
neg_logl_true_std, std_dev = ibs_sampler(true_params; num_reps=10, additional_output="std")
println("With std dev - Neg-LL: $(round(neg_logl_true_std, digits=4)), Std: $(round(std_dev, digits=4))")

# Full estimation result
result = ibs_sampler(true_params; num_reps=20, additional_output="full")
println("\nFull result:")
println("  Negative log-likelihood: $(round(result.neg_logl, digits=4))")
println("  Standard deviation: $(round(result.neg_logl_std, digits=4))")
println("  Exit flag: $(result.exit_flag) ($(result.message))")
println("  Elapsed time: $(round(result.elapsed_time, digits=3)) seconds")
println("  Avg samples per trial: $(round(result.num_samples_per_trial, digits=1))")
println("  Function evaluations: $(result.fun_count)")

# =============================================================================
# Step 4: Compare Different Parameter Sets
# =============================================================================
println("\n4. Comparing different parameter sets...")

# Test parameters: true params, slightly perturbed params, and random params
test_cases = [
    ("True parameters", true_params),
    ("Perturbed (+10%)", true_params .* 1.1),
    ("Perturbed (-10%)", true_params .* 0.9),
    ("Different d1", [true_params[1]*2, true_params[2:end]...]),
    ("Different th1", [true_params[1], true_params[2], true_params[3]*1.5, true_params[4:end]...]),
    ("Random parameters", [1e-5, 1e-5, 0.2, 0.2, 200.0, 200.0])
]

println("\nParameter set comparison:")
println("Name                | Neg-LL    | Std Dev   | Rel. to True")
println("-" ^ 55)

baseline_ll = neg_logl_true

for (name, params) in test_cases
    try
        neg_ll, std_dev = ibs_sampler(params; num_reps=10, additional_output="std")
        relative_diff = neg_ll - baseline_ll
        
        println("$(rpad(name, 18)) | $(rpad(round(neg_ll, digits=3), 8)) | $(rpad(round(std_dev, digits=3), 8)) | $(round(relative_diff, digits=3))")
    catch e
        println("$(rpad(name, 18)) | ERROR: $(typeof(e).name)")
    end
end

# =============================================================================
# Step 5: Parameter Sensitivity Analysis
# =============================================================================
println("\n5. Parameter sensitivity analysis...")

# Test how log-likelihood changes with one parameter
println("\nSensitivity of d1 parameter:")
d1_values = [4e-6, 6e-6, 8e-6, 10e-6, 12e-6]
println("d1 value    | Neg-LL    | Diff from true")
println("-" ^ 35)

for d1 in d1_values
    test_params = [d1, true_params[2:end]...]
    neg_ll = ibs_sampler(test_params; num_reps=5)  # Fewer reps for speed
    diff = neg_ll - baseline_ll
    println("$(rpad(d1, 10)) | $(rpad(round(neg_ll, digits=3), 8)) | $(round(diff, digits=3))")
end

# =============================================================================
# Step 6: Trial Weighting Example
# =============================================================================
println("\n6. Trial weighting example...")

# Weight trials differently (e.g., weight later trials more)
trial_weights = collect(1:n_trials) / n_trials  # Linear increasing weights
neg_ll_weighted = ibs_sampler(true_params; num_reps=10, trial_weights=trial_weights)

println("Unweighted neg-LL: $(round(neg_logl_true, digits=4))")
println("Weighted neg-LL:   $(round(neg_ll_weighted, digits=4))")
println("Difference:        $(round(neg_ll_weighted - neg_logl_true, digits=4))")

# =============================================================================
# Summary
# =============================================================================
println("=" ^ 60)
println("Demo Summary:")
println("=" ^ 60)
println("✓ Generated synthetic data with $(n_trials) trials")
println("✓ Created IBS sampler with lapse model")
println("✓ Estimated log-likelihood for true parameters: $(round(neg_logl_true, digits=4))")
println("✓ Compared multiple parameter sets")
println("✓ Performed sensitivity analysis")
println("✓ Demonstrated trial weighting")
println("\nKey insights:")
println("• IBS provides unbiased likelihood estimates when exit_flag = 0")
println("• Parameter perturbations increase negative log-likelihood")
println("• Standard deviation quantifies estimation uncertainty")
println("• Trial weighting allows focusing on specific trials")
println("\nThe IBS sampler is ready for parameter fitting with optimization algorithms!") 