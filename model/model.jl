# model.jl - Main model function for two-stage decision making

using Random
using Distributions
include("utils.jl")
include("tree.jl")
include("beliefs.jl")
include("policy.jl")
include("sampling.jl")

"""
    SamplingStep

Represents a single sampling step in the decision process.
"""
struct SamplingStep
    stage::Int                    # 1 or 2
    step_idx::Int                # Step index within stage
    node_observed::Int            # Node that was observed
    observation::Float64         # Observed reward value
    policy_posterior::Vector{Float64}  # Policy posterior after this step
    reward_estimates::Vector{Float64}  # Mean reward estimates
    reward_variances::Vector{Float64}  # Reward variances
    entropy::Float64             # Entropy of policy posterior
end

"""
    ModelParameters

Parameters for the two-stage decision model.
"""
struct ModelParameters
    mu0::Float64                 # Prior mean
    tau0::Float64                # Prior std
    sigma::Float64               # Observation noise
    gamma::Float64               # Depth discount factor
    nsamples_lik::Int            # MC samples for likelihood (if using MC)
    H_thresh1::Float64           # Entropy threshold for stage 1
    H_thresh2::Float64           # Entropy threshold for stage 2
    Tsoft::Float64               # Temperature parameter
    eps_nodesel::Float64         # Exploration epsilon
    dt::Float64                  # Time step for posterior update
    alpha::Float64               # Update rate constant
    p_stay::Float64              # Persistence probability
    lambda::Float64              # Distance decay parameter
    use_genz::Bool               # Use Genz method (true) or MC (false)
end

function ModelParameters(;
    mu0=0.0,
    tau0=1.0,
    sigma=0.6,
    gamma=1.0,
    nsamples_lik=3000,
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
    ModelParameters(mu0, tau0, sigma, gamma, nsamples_lik, H_thresh1, H_thresh2,
                   Tsoft, eps_nodesel, dt, alpha, p_stay, lambda, use_genz)
end

"""
    ModelResult

Result of running the two-stage decision model.
"""
struct ModelResult
    choice1::Int                 # First stage choice (1=L, 2=R)
    choice2::Int                 # Second stage choice (1=first option, 3=second option)
    rt1::Int                     # Reaction time for stage 1 (number of observations)
    rt2::Int                     # Reaction time for stage 2 (number of observations)
    trajectory::Vector{SamplingStep}  # Complete sampling trajectory
    final_beliefs::NodeBeliefs   # Final node beliefs
end

"""
    run_model(rewards, params; rngseed=nothing)

Main model function that runs the two-stage decision process.

Args:
    rewards: Vector of 6 true reward values [L, R, LL, LR, RL, RR]
    params: ModelParameters struct
    rngseed: Optional random seed

Returns:
    ModelResult with choices, reaction times, and trajectory
"""
function run_model(rewards::Vector{Float64}, params::ModelParameters; rngseed::Union{Int,Nothing}=nothing)
    
    if rngseed !== nothing
        Random.seed!(rngseed)
    end
    
    if length(rewards) != 6
        error("rewards must be a vector of length 6: [L, R, LL, LR, RL, RR]")
    end
    
    tree = make_tree_2layer_full()
    nπ = size(tree.paths, 1)
    nb = NodeBeliefs(tree.nnodes; mu0=params.mu0, tau0=params.tau0, sigma=params.sigma)
    
    trajectory = SamplingStep[]
    D = node_distance_matrix(tree)
    
    # ========== Stage 1 ==========
    feasible1 = collect(1:nπ)
    Π = zeros(nπ); Π[feasible1] .= 1/length(feasible1)
    last_node = nothing
    step_idx = 0
    
    # Record initial state
    push!(trajectory, SamplingStep(
        1, 0, 0, 0.0, copy(Π), copy(nb.mu), copy(nb.var), entropy(normalize_prob(Π[feasible1]))
    ))
    
    while true
        step_idx += 1
        
        # Choose node to observe
        node = sample_node_persist_prox(
            tree, Π, feasible1;
            last_node=last_node, D=D,
            p_stay=params.p_stay, λ=params.lambda, ε=params.eps_nodesel
        )
        last_node = node
        
        # Observe reward
        observation = observe_reward(node, rewards[node], params.sigma)
        
        # Update belief
        update_belief!(nb, node, observation)
        
        # Compute policy likelihood
        if params.use_genz
            lik = policy_likelihood_optimal_Genz(tree, nb; γ=params.gamma)
        else
            lik = policy_likelihood_optimal_MC(tree, nb; γ=params.gamma, nsamples=params.nsamples_lik)
        end
        
        # Compute target posterior
        Π_target = posterior_over_paths(lik; Tsoft=params.Tsoft)
        
        # Update posterior with time step
        Π = update_posterior_with_timestep(Π, Π_target; Δt=params.dt, α=params.alpha)
        
        # Record step
        Πf = normalize_prob([Π[p] for p in feasible1])
        ent = entropy(Πf)
        push!(trajectory, SamplingStep(
            1, step_idx, node, observation, copy(Π), copy(nb.mu), copy(nb.var), ent
        ))
        
        # Check termination
        if ent ≤ params.H_thresh1
            break
        end
    end
    
    # Make stage 1 decision
    π1 = argmax(Π)
    a1 = tree.paths[π1, 1]
    rt1 = step_idx
    
    # ========== Stage 2 ==========
    feasible2 = [p for p in feasible1 if tree.paths[p,1] == a1]
    isempty(feasible2) && (feasible2 = [p for p in 1:nπ if tree.paths[p,1] == a1])
    allowed2 = feasible_nodes(tree, feasible2)
    
    Π2 = zeros(nπ); Π2[feasible2] .= 1/length(feasible2)
    step_idx = 0
    
    # Record initial state of stage 2
    push!(trajectory, SamplingStep(
        2, 0, 0, 0.0, copy(Π2), copy(nb.mu), copy(nb.var), entropy(normalize_prob(Π2[feasible2]))
    ))
    
    while true
        step_idx += 1
        
        # Choose node to observe
        node = sample_node_persist_prox(
            tree, Π2, feasible2;
            allowed_nodes=allowed2, last_node=last_node, D=D,
            p_stay=params.p_stay, λ=params.lambda, ε=params.eps_nodesel
        )
        last_node = node
        
        # Observe reward
        observation = observe_reward(node, rewards[node], params.sigma)
        
        # Update belief
        update_belief!(nb, node, observation)
        
        # Compute policy likelihood
        if params.use_genz
            lik2 = policy_likelihood_optimal_Genz(tree, nb; γ=params.gamma)
        else
            lik2 = policy_likelihood_optimal_MC(tree, nb; γ=params.gamma, nsamples=params.nsamples_lik)
        end
        
        # Compute target posterior (restricted to feasible paths)
        Π_all_target = posterior_over_paths(lik2; Tsoft=params.Tsoft)
        Π2_target = zeros(nπ)
        Π2_target[feasible2] = normalize_prob([Π_all_target[p] for p in feasible2])
        
        # Update posterior with time step
        Π2 = update_posterior_with_timestep(Π2, Π2_target; Δt=params.dt, α=params.alpha)
        
        # Record step
        Πf2 = normalize_prob([Π2[p] for p in feasible2])
        ent2 = entropy(Πf2)
        push!(trajectory, SamplingStep(
            2, step_idx, node, observation, copy(Π2), copy(nb.mu), copy(nb.var), ent2
        ))
        
        # Check termination
        if ent2 ≤ params.H_thresh2
            break
        end
    end
    
    # Make stage 2 decision
    π2 = argmax(Π2)
    a2 = tree.paths[π2, 2]
    rt2 = step_idx
    
    # choice2 is simply the second stage action (1 or 2)
    # This represents the choice within the selected branch
    choice2 = a2
    
    return ModelResult(a1, choice2, rt1, rt2, trajectory, nb)
end

# Convenience function with default parameters
function run_model(rewards::Vector{Float64}; 
                  mu0=0.0, tau0=1.0, sigma=0.6, gamma=1.0,
                  H_thresh1=log(4), H_thresh2=log(2),
                  Tsoft=1.0, eps_nodesel=0.05, dt=0.01, alpha=1.0,
                  p_stay=0.75, lambda=1.2, use_genz=true,
                  rngseed::Union{Int,Nothing}=nothing)
    params = ModelParameters(;
        mu0=mu0, tau0=tau0, sigma=sigma, gamma=gamma,
        H_thresh1=H_thresh1, H_thresh2=H_thresh2,
        Tsoft=Tsoft, eps_nodesel=eps_nodesel, dt=dt, alpha=alpha,
        p_stay=p_stay, lambda=lambda, use_genz=use_genz
    )
    run_model(rewards, params; rngseed=rngseed)
end

