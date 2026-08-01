# Two-Stage Decision Model - Modular Implementation

This directory contains a modular implementation of the two-stage decision-making model.

## File Structure

- **`utils.jl`**: Utility functions (normalize_prob, drawcat, entropy)
- **`tree.jl`**: Decision tree structure and path computation
- **`beliefs.jl`**: Node reward beliefs (Normal-Normal conjugate updates)
- **`policy.jl`**: Policy likelihood computation (Genz and MC methods)
- **`sampling.jl`**: Node sampling strategies (persistence and proximity)
- **`model.jl`**: Main model function that orchestrates the decision process

## Usage

### Basic Usage

```julia
include("model.jl")

# Define true rewards [L, R, LL, LR, RL, RR]
rewards = [1.0, 0.0, 3.0, 1.0, 4.0, 2.0]

# Create parameters
params = ModelParameters(;
    mu0=0.0,
    tau0=1.0,
    sigma=0.6,
    gamma=1.0,
    H_thresh1=log(4),
    H_thresh2=log(2),
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

# Access results
println("Choice 1: ", result.choice1)  # 1=L, 2=R
println("Choice 2: ", result.choice2)  # 1 or 2 (within branch)
println("RT1: ", result.rt1)            # Number of observations in stage 1
println("RT2: ", result.rt2)           # Number of observations in stage 2

# Access trajectory
for step in result.trajectory
    println("Stage ", step.stage, ", Node ", step.node_observed, 
            ", Observation ", step.observation)
end
```

### Convenience Function

You can also use the convenience function with keyword arguments:

```julia
result = run_model(rewards; 
                  mu0=0.0, tau0=1.0, sigma=0.6,
                  gamma=1.0, H_thresh1=log(4), H_thresh2=log(2),
                  Tsoft=1.0, eps_nodesel=0.05, dt=0.01, alpha=1.0,
                  p_stay=0.75, lambda=1.2, use_genz=true,
                  rngseed=69)
```

## Model Parameters

- **`mu0`**: Prior mean for node rewards (default: 0.0)
- **`tau0`**: Prior standard deviation (default: 1.0)
- **`sigma`**: Observation noise standard deviation (default: 0.6)
- **`gamma`**: Depth discount factor (default: 1.0)
- **`H_thresh1`**: Entropy threshold for stage 1 termination (default: log(4))
- **`H_thresh2`**: Entropy threshold for stage 2 termination (default: log(2))
- **`Tsoft`**: Temperature parameter for policy posterior (default: 1.0)
- **`eps_nodesel`**: Exploration epsilon for node selection (default: 0.05)
- **`dt`**: Time step for posterior updating (default: 0.01)
- **`alpha`**: Update rate constant (default: 1.0)
- **`p_stay`**: Persistence probability (default: 0.75)
- **`lambda`**: Distance decay parameter (default: 1.2)
- **`use_genz`**: Use Genz method (true) or MC (false) for likelihood (default: true)

## Return Value

The `run_model` function returns a `ModelResult` struct containing:

- **`choice1`**: First stage choice (1 = L, 2 = R)
- **`choice2`**: Second stage choice (1 or 2, within the selected branch)
- **`rt1`**: Reaction time for stage 1 (number of observations)
- **`rt2`**: Reaction time for stage 2 (number of observations)
- **`trajectory`**: Vector of `SamplingStep` structs containing:
  - `stage`: Stage number (1 or 2)
  - `step_idx`: Step index within stage
  - `node_observed`: Node that was observed
  - `observation`: Observed reward value
  - `policy_posterior`: Policy posterior after this step
  - `reward_estimates`: Mean reward estimates
  - `reward_variances`: Reward variances
  - `entropy`: Entropy of policy posterior
- **`final_beliefs`**: Final `NodeBeliefs` structure with updated means and variances

## Testing

Run the test script:

```julia
include("test_model.jl")
```

This will run a single trial and print detailed results.

