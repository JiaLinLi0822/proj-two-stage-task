# test_model.jl - Test script for the modular model

include("model.jl")

# Test with example rewards
rewards = [1.0, 0.0, 3.0, 1.0, 4.0, 2.0]  # [L, R, LL, LR, RL, RR]

# Create parameters
params = ModelParameters(;
    mu0=0.0,
    tau0=1.0,
    sigma=0.6,
    gamma=1.0,
    H_thresh1=0.85,
    H_thresh2=0.45,
    Tsoft=1.0,
    eps_nodesel=0.05,
    dt=0.01,
    alpha=1.0,
    p_stay=0.75,
    lambda=1.2,
    use_genz=true
)

# Run model
result = run_model(rewards, params; rngseed=69)

# Print results
println("=" ^ 60)
println("Model Results")
println("=" ^ 60)
println("Choice 1 (Stage 1): ", result.choice1, " (1=L, 2=R)")
println("Choice 2 (Stage 2): ", result.choice2, " (1=first option, 2=second option)")
println("RT1 (Stage 1): ", result.rt1, " observations")
println("RT2 (Stage 2): ", result.rt2, " observations")
println("Total observations: ", result.rt1 + result.rt2)
println("\nTrajectory summary:")
println("  Stage 1 steps: ", count(s -> s.stage == 1, result.trajectory))
println("  Stage 2 steps: ", count(s -> s.stage == 2, result.trajectory))
println("\nFinal reward estimates:")
for (i, (mu, var)) in enumerate(zip(result.final_beliefs.mu, result.final_beliefs.var))
    node_names = ["L", "R", "LL", "LR", "RL", "RR"]
    println("  Node ", node_names[i], ": μ = ", round(mu, digits=3), ", σ² = ", round(var, digits=3))
end

println("\nFirst 5 sampling steps:")
for (i, step) in enumerate(result.trajectory[1:min(5, length(result.trajectory))])
    println("  Step ", i, " (Stage ", step.stage, ", idx ", step.step_idx, "):")
    println("    Node: ", step.node_observed, ", Observation: ", round(step.observation, digits=3))
    println("    Entropy: ", round(step.entropy, digits=3))
end

