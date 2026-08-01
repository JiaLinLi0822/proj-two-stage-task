# policy.jl - Policy likelihood and posterior computation

using Distributions
using LinearAlgebra
using MvNormalCDF

function policy_likelihood_optimal_Genz(tree::Tree2LayerFull, nb::NodeBeliefs; γ=1.0, m::Int=5000)
    nπ = size(tree.paths, 1)
    A  = path_incidence_matrix(tree; γ=γ)

    # 节点后验：R ~ N(mu_R, Σ_R)
    μR = nb.mu
    ΣR = Diagonal(nb.var)          # 6×6

    # 路径价值：V = A*R  =>  V ~ N(μV, ΣV)
    μV = A * μR                    # 4×1
    ΣV = A * ΣR * A'               # 4×4

    lik = zeros(Float64, nπ)

    for k in 1:nπ
        # 构造差值矩阵 D_k: 每行是 e_k - e_j, j ≠ k
        others = setdiff(1:nπ, [k])
        D = zeros(Float64, length(others), nπ)
        for (row, j) in enumerate(others)
            D[row, k] =  1.0
            D[row, j] = -1.0
        end

        μY = D * μV                 # 3×1
        ΣY = D * ΣV * D'            # 3×3

        a = zeros(length(others))   # 下界 0
        b = fill(Inf, length(others))

        # Genz / quasi-MC 计算 P( Y ∈ [0,∞)^3 )
        (p, err) = mvnormcdf(μY, ΣY, a, b; m=m)
        lik[k] = p
    end

    lik .+= 1e-12
    lik ./= sum(lik)
    return lik
end

function policy_likelihood_optimal_MC(tree::Tree2LayerFull, nb::NodeBeliefs; γ=1.0, nsamples=4000)
    nπ = size(tree.paths,1)
    A  = path_incidence_matrix(tree; γ=γ)
    counts = zeros(Int, nπ)
    σ = sqrt.(max.(nb.var, 1e-12))
    for _ in 1:nsamples
        Rdraw = rand.(Normal.(nb.mu, σ))          # draw node values
        V = A * collect(Rdraw)                    # path values
        pwin = findmax(V)[2]                      # winner path
        counts[pwin] += 1
    end
    lik = counts ./ max(nsamples,1)
    lik = (lik .+ 1e-12); lik ./= sum(lik)
    lik
end

function posterior_over_paths(lik::Vector{Float64}; Tsoft::Float64=1.0)
    """
    Compute policy posterior from likelihood using temperature.
    Posterior ∝ exp(log(lik) / Tsoft)
    """
    logu = log.(lik .+ 1e-12) ./ max(Tsoft, 1e-12)
    u = exp.(logu .- maximum(logu)) # stabilize
    normalize_prob(u)
end

function update_posterior_with_timestep(Π_current::Vector{Float64}, Π_target::Vector{Float64};
                                       Δt::Float64=0.001, α::Float64=1.0)
    """
    Update posterior using time step Δt.
    Π(t+Δt) = Π(t) + Δt * α * (Π_target - Π(t))
    """
    Π_new = Π_current .+ Δt * α .* (Π_target .- Π_current)
    normalize_prob(Π_new)
end

