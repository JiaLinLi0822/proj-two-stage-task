# beliefs.jl - Node reward beliefs (Normal-Normal conjugate)

using Distributions

# -------- node reward posteriors (Normal–Normal) --------
mutable struct NodeBeliefs
    mu::Vector{Float64}
    var::Vector{Float64}
    mu0::Float64
    tau0sq::Float64
    sigmasq::Float64
    sumx::Vector{Float64}
    cnt::Vector{Int}
end

function NodeBeliefs(nnodes; mu0=0.0, tau0=1.0, sigma=0.6)
    NodeBeliefs(fill(mu0, nnodes), fill(tau0^2, nnodes), mu0, tau0^2, sigma^2,
                zeros(nnodes), zeros(Int, nnodes))
end

@inline function posterior_params(mu0, tau0sq, sigmasq, sumx, cnt)
    τpostsq = 1.0 / (1/tau0sq + cnt/sigmasq)
    μpost   = τpostsq * (mu0/tau0sq + sumx/sigmasq)
    μpost, τpostsq
end

function update_belief!(nb::NodeBeliefs, node_idx::Int, observation::Float64)
    """
    Update node belief after observing a reward sample.
    
    Args:
        nb: NodeBeliefs structure
        node_idx: Index of the node (1-6)
        observation: Observed reward value
    """
    nb.sumx[node_idx] += observation
    nb.cnt[node_idx] += 1
    nb.mu[node_idx], nb.var[node_idx] = posterior_params(
        nb.mu0, nb.tau0sq, nb.sigmasq,
        nb.sumx[node_idx], nb.cnt[node_idx]
    )
end

function observe_reward(node_idx::Int, true_reward::Float64, sigma::Float64)
    """
    Generate an observation from the true reward with noise.
    
    Args:
        node_idx: Node index (not used, but kept for consistency)
        true_reward: True reward value
        sigma: Observation noise standard deviation
    
    Returns:
        observation: Noisy reward observation
    """
    rand(Normal(true_reward, sigma))
end

