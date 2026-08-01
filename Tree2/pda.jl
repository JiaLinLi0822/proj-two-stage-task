using Statistics
using LinearAlgebra
using Dates

# ---------- Multivariate Gaussian KDE ----------
@inline _scotts_factor(n::Real, d::Integer) = float(n)^(-1.0 / (d + 4))
@inline _silverman_factor(n::Real, d::Integer) = (float(n) * (d + 2) / 4.0)^(-1.0 / (d + 4))

# ---------- 2D KDE (Gaussian) ----------
struct KDE2D_Gaussian
    X::Matrix{Float64}    
    H::Matrix{Float64} 
    L::LowerTriangular{Float64,Matrix{Float64}} 
    logRT::Bool
    eps_floor::Float64
end

# ---------- Mixed SPDF over pairs (c1,c2) ----------
struct Mixed2DSPDF
    prior:: Float64
    kde::KDE2D_Gaussian
    eps_floor::Float64
end


function fit_kde2d_gaussian(s1::Vector{Float64}, s2::Vector{Float64}; logRT::Bool=true, 
    bw_rule::Symbol=:silverman, eps_floor::Float64=1e-16, bw_scale::Float64=1.0)

    @assert length(s1) == length(s2)
    n = length(s1)
    d = 2

    if n <= d+1
        X = zeros(0, d)
        H = Matrix{Float64}(I, d, d)
        L = LowerTriangular(Matrix{Float64}(I, d, d))
        return KDE2D_Gaussian(X, H, L, logRT, eps_floor)
    end

    x1 = copy(s1); x2 = copy(s2)
    if logRT
        x1 = log.(x1 .+ 1.0)
        x2 = log.(x2 .+ 1.0)
    end

    X = hcat(x1, x2)
    μ = vec(mean(X, dims=1))
    XC = X .- μ'
    Σ  = (XC' * XC) / (n - 1)

    # if the variance in either dimension is smaller than 1e-6, return a very small eps_floor
    if Σ[1,1] < 1e-6 || Σ[2,2] < 1e-6
        H = Matrix{Float64}(I, d, d)
        L = LowerTriangular(Matrix{Float64}(I, d, d))
        return KDE2D_Gaussian(X, H, L, logRT, eps_floor)
    end

    factor = (bw_rule == :scott)      ? _scotts_factor(n, d) :
             (bw_rule == :silverman)  ? _silverman_factor(n, d) :
             error("Unknown bw_rule = $bw_rule (use :scott or :silverman)")

    H = (factor * bw_scale)^2 .* Σ
    L = cholesky(Symmetric(H)).L

    return KDE2D_Gaussian(X, H, L, logRT, eps_floor)

end


function logpdf(k::KDE2D_Gaussian, v1::Float64, v2::Float64)
    
    n = size(k.X, 1)
    d = size(k.X, 2)

    if n <= d+1
        return log(k.eps_floor)
    end
    
    if k.logRT
        z1 = log(v1 + 1.0)
        z2 = log(v2 + 1.0)
    else
        z1 = v1
        z2 = v2
    end

    log_denom = (d/2)*log(2π) + sum(log.(diag(k.L)))
    L = k.L
    ℓ11, ℓ21, ℓ22 = L[1,1], L[2,1], L[2,2]
    
    # Check for zero diagonal elements to avoid division by zero
    if ℓ11 <= 1e-6 || ℓ22 <= 1e-6
        return log(k.eps_floor)
    end
    
    invℓ11, invℓ22 = 1/ℓ11, 1/ℓ22

    smax = -Inf
    tmp = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        u1 = z1 - k.X[i,1]
        u2 = z2 - k.X[i,2]
        t1 = u1 * invℓ11
        t2 = (u2 - ℓ21 * t1) * invℓ22
        e  = 0.5 * (t1*t1 + t2*t2)
        li = -(log_denom) - e
        tmp[i] = li
        if li > smax; smax = li; end
    end

    acc = 0.0
    @inbounds @simd for i in 1:n
        acc += exp(tmp[i] - smax)
    end
    logdens = smax + log(acc) - log(n)

    if !isfinite(logdens)
        logdens = log(k.eps_floor)
    end

    if k.logRT
        logdens -= log(v1 + 1)
        logdens -= log(v2 + 1)
    end

    return logdens
end


function build_mixed2d_spdf(results::AbstractVector, trial::Trial;
                            pairs::Vector{Tuple{Int,Int}},
                            bw_rule::Symbol=:silverman, 
                            logRT::Bool=true,
                            eps_floor::Float64=1e-16,
                            bw_scale::Float64=1.0)

    J = length(results)
    if J <= 2
        return Mixed2DSPDF(0.0, nothing, eps_floor)
    end

    counts   = Dict{Tuple{Int,Int},Int}(p => 0 for p in pairs)
    buckets1 = Dict{Tuple{Int,Int},Vector{Float64}}(p => Float64[] for p in pairs)
    buckets2 = Dict{Tuple{Int,Int},Vector{Float64}}(p => Float64[] for p in pairs)

    @inbounds for i in 1:J
        p = (results[i].choice1, results[i].choice2)
        if haskey(counts, p)
            counts[p] += 1
            push!(buckets1[p], results[i].rt1)
            push!(buckets2[p], results[i].rt2)
        end
    end

    p = (trial.choice1, trial.choice2)
    prior = counts[p] / J

    kde = fit_kde2d_gaussian(buckets1[p], buckets2[p];
                            logRT=logRT, bw_rule=bw_rule, eps_floor=eps_floor, bw_scale=bw_scale)
    return Mixed2DSPDF(prior, kde, eps_floor)

end

function mixed2d_logpdf(spdf::Mixed2DSPDF, trial::Trial, lambda::Float64=1.0)

    if spdf.kde === nothing
        ll = log(spdf.eps_floor)
        used_eps = true
    else
        logprior = log(spdf.prior)
        logdens = logpdf(spdf.kde, trial.rt1, trial.rt2)
        ll = logprior + lambda * logdens
        # Check if we're using eps_floor (either prior is 0 or logdens hit the floor)
        used_eps = (spdf.prior <= 0.0) || (ll <= log(spdf.eps_floor) + 1e-10)
    end

    ll = max(ll, log(spdf.eps_floor))

    return ll, used_eps
end

# ----------------- Adaptive simulation for PDA -----------------
function pda_sampler(trial::Trial, params::Vector{Float64}, model_function::Function; 
                                   min_sims::Int=1000,
                                   min_matching::Int=100,
                                   max_sims::Int=1000000)
    """
    Generate samples adaptively: first simulate at least min_sims times, then check
    if we have at least min_matching samples matching the target choice pair.
    If not, continue simulating until reaching min_matching or max_sims.
    
    Args:
        trial: Target trial with specific choice pair
        params: Model parameters
        model_function: Model function to use for simulation (e.g., model1, model6, etc.)
        min_sims: Minimum number of simulations to perform (default: 1000)
        min_matching: Minimum number of matching samples required (default: 100)
        max_sims: Maximum number of simulations to prevent infinite loops
    
    Returns:
        samples: Vector of all simulated trials
    """

    target_pair = (trial.choice1, trial.choice2)
    samples = Trial[]
    total_sims = 0
    matching_sims = 0   
    
    # First phase: simulate at least min_sims times
    while total_sims < min_sims && total_sims < max_sims
        sim_result = model_function(params, trial.rewards)
        total_sims += 1
        
        sim_trial = Trial(
            trial.wid,
            trial.rewards,
            sim_result.choice1,
            sim_result.choice2,
            sim_result.rt1,
            sim_result.rt2,
            trial.path
        )

        push!(samples, sim_trial)
        
        if (sim_trial.choice1, sim_trial.choice2) == target_pair
            matching_sims += 1
        end
    end
    
    # Second phase: continue simulating if we don't have enough matching samples
    while matching_sims < min_matching && total_sims < max_sims
        sim_result = model_function(params, trial.rewards)
        total_sims += 1
        
        sim_trial = Trial(
            trial.wid,
            trial.rewards,
            sim_result.choice1,
            sim_result.choice2,
            sim_result.rt1,
            sim_result.rt2,
            trial.path
        )

        push!(samples, sim_trial)
        
        if (sim_trial.choice1, sim_trial.choice2) == target_pair
            matching_sims += 1
            if matching_sims >= min_matching
                break
            end
        end
    end
    
    return samples
end


function compute_pda_joint_choice_rt(sim_results::Vector{Trial},
                                    tr::Trial,
                                    t_values::Vector{Float64};
                                    logRT::Bool=false,
                                    bw_rule::Symbol=:silverman,
                                    bw_scale::Float64=1.0,
                                    min_samples::Int=10)

    c1_star = tr.choice1
    c2_star = tr.choice2

    dt = t_values[2] - t_values[1]
    eps_floor = 1e-16

    buckets1 = Dict{Tuple{Int,Int}, Vector{Float64}}()
    buckets2 = Dict{Tuple{Int,Int}, Vector{Float64}}()
    counts   = Dict{Tuple{Int,Int}, Int}()

    for r in sim_results
        p = (r.choice1, r.choice2)
        if !haskey(buckets1, p)
            buckets1[p] = Float64[]
            buckets2[p] = Float64[]
            counts[p]   = 0
        end
        push!(buckets1[p], r.rt1)
        push!(buckets2[p], r.rt2)
        counts[p] += 1
    end

    pairs   = collect(keys(counts))
    J_total = length(sim_results)

    # pair-level prior: π_p = p(c1,c2)
    priors = Dict{Tuple{Int,Int}, Float64}()
    for (p, c) in counts
        priors[p] = c / J_total
    end

    prior_c1 = sum(priors[p] for p in pairs if p[1] == c1_star)
    prior_c2 = sum(priors[p] for p in pairs if p[2] == c2_star)

    all_rt1 = [r.rt1 for r in sim_results]
    all_rt2 = [r.rt2 for r in sim_results]

    rt1_int_range = range(minimum(all_rt1), stop=maximum(all_rt1), length=100)
    rt2_int_range = range(minimum(all_rt2), stop=maximum(all_rt2), length=100)

    Δrt1 = rt1_int_range[2] - rt1_int_range[1]
    Δrt2 = rt2_int_range[2] - rt2_int_range[1]

    pdf_rt1 = zeros(Float64, length(t_values))  # for p(c1*, rt1)
    pdf_rt2 = zeros(Float64, length(t_values))  # for p(c2*, rt2)

    for p in pairs
        c1, c2 = p
        rt1_samples = buckets1[p]
        rt2_samples = buckets2[p]

        if length(rt1_samples) < min_samples
            continue
        end

        prior_p = priors[p]           # p(c1,c2)
        prior_p == 0.0 && continue

        kde_p = fit_kde2d_gaussian(rt1_samples, rt2_samples;
                                   logRT=logRT,
                                   bw_rule=bw_rule,
                                   eps_floor=eps_floor,
                                   bw_scale=bw_scale)

        if c1 == c1_star
            for (i, t1) in enumerate(t_values)
                s = 0.0
                for t2 in rt2_int_range
                    s += exp(logpdf(kde_p, t1, t2))
                end
                pdf_rt1[i] += prior_p * s * Δrt2
            end
        end

        if c2 == c2_star
            for (j, t2) in enumerate(t_values)
                s = 0.0
                for t1 in rt1_int_range
                    s += exp(logpdf(kde_p, t1, t2))
                end
                pdf_rt2[j] += prior_p * s * Δrt1
            end
        end
    end

    mass_c1 = sum(pdf_rt1) * dt
    if mass_c1 > 0 && prior_c1 > 0
        pdf_rt1 .*= (prior_c1 / mass_c1)
    end

    mass_c2 = sum(pdf_rt2) * dt
    if mass_c2 > 0 && prior_c2 > 0
        pdf_rt2 .*= (prior_c2 / mass_c2)
    end

    cdf_rt1 = cumsum(pdf_rt1) * dt
    cdf_rt2 = cumsum(pdf_rt2) * dt

    cdf_rt1 = min.(cdf_rt1, prior_c1)
    cdf_rt2 = min.(cdf_rt2, prior_c2)

    return (
        pdf_rt1  = pdf_rt1,   # ≈ p_PDA(c1 = c1*, rt1 = t)
        cdf_rt1  = cdf_rt1,   # ≈ P_PDA(RT1 ≤ t, c1 = c1*)
        pdf_rt2  = pdf_rt2,   # ≈ p_PDA(c2 = c2*, rt2 = t)
        cdf_rt2  = cdf_rt2,   # ≈ P_PDA(RT2 ≤ t, c2 = c2*)
        prior_c1 = prior_c1,  # ≈ p_PDA(c1 = c1*)
        prior_c2 = prior_c2,  # ≈ p_PDA(c2 = c2*)
    )
end