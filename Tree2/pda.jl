# pda.jl
# Probability Density Approximation
# - Mixed SPDF over pairs (c1,c2)
# - 2D KDE (product Epanechnikov)
# - 2D KDE (Gaussian)
# - Log-transform with shift and Jacobian correction
# - Eps floor

using Statistics
using LinearAlgebra
using Dates

########################################################
################ Utility Functions #####################
########################################################

# ---------- 2D Epanechnikov product kernel ----------
@inline _epan(u::Float64) = (abs(u) <= 1.0) ? 0.75*(1 - u^2) : 0.0
@inline function _epan2(u1::Float64, u2::Float64)
    (abs(u1) <= 1.0 && abs(u2) <= 1.0) ? 0.75*(1 - u1^2) * 0.75*(1 - u2^2) : 0.0
end

# ---------- Multivariate Gaussian KDE ----------
@inline _scotts_factor(n::Real, d::Integer) = float(n)^(-1.0 / (d + 4))
@inline _silverman_factor(n::Real, d::Integer) = (float(n) * (d + 2) / 4.0)^(-1.0 / (d + 4))


#######################################################
####################### Structs #######################
#######################################################

# ---------- 2D KDE (product Epanechnikov) ----------
struct KDE2D
    x1::Vector{Float64}   # transformed rt1
    x2::Vector{Float64}   # transformed rt2
    h1::Float64
    h2::Float64
    logRT::Bool
    eps_floor::Float64
end

# ---------- 2D KDE (Gaussian) ----------
struct KDE2D_Gaussian
    X::Matrix{Float64}         # n×d transformed samples (rows = obs)
    H::Matrix{Float64}         # 2×2 bandwidth matrix
    L::LowerTriangular{Float64,Matrix{Float64}} # chol of H
    logRT::Bool
    eps_floor::Float64
end

# ---------- Mixed SPDF over pairs (c1,c2) ----------
struct Mixed2DSPDF
    prior:: Float64
    kde::Any   # KDE2D or KDE2D_Gaussian
    eps_floor::Float64
end



#######################################################
################## 2D Product Kernel ##################
#######################################################

#------------- Silverman bandwidth (1D) -------------
function silverman_bandwidth_1d(xs::AbstractVector{<:Real}; min_h::Float64=1e-3)
    n = length(xs)
    if n <= 1
        return max(1.0, min_h)
    end
    μ = mean(xs)
    sd = sqrt(mean((xs .- μ).^2))
    q75 = quantile(xs, 0.75); q25 = quantile(xs, 0.25)
    iqr = q75 - q25
    h = 0.9 * min(sd, iqr/1.34) * n^(-1/5)
    if !isfinite(h) || h <= 0
        h = 1.0
    end
    return max(h, min_h)
end

#--------- Fit 2D KDE (product Epanechnikov) ---------
function fit_kde2d_product(s1::Vector{Float64}, s2::Vector{Float64}; logRT::Bool=true, min_h::Float64=1e-3, eps_floor::Float64=1e-16)
    
    @assert length(s1) == length(s2)
    n = length(s1)
    if n == 0
        return KDE2D(Float64[], Float64[], 1.0, 1.0, logRT, eps_floor)
    end

    x1 = copy(s1); x2 = copy(s2)
    if logRT
        x1 = log.(x1 .+ 1.0)
        x2 = log.(x2 .+ 1.0)
    end

    h1 = silverman_bandwidth_1d(x1; min_h=min_h)
    h2 = silverman_bandwidth_1d(x2; min_h=min_h)
    KDE2D(x1, x2, h1, h2, logRT, eps_floor)
end



function logpdf(k::KDE2D, v1::Float64, v2::Float64)
    n = length(k.x1)
    if n == 0
        return log(k.eps_floor)
    end

    if k.logRT
        z1 = log(v1 + 1.0)
        z2 = log(v2 + 1.0)
    else
        z1 = v1
        z2 = v2
    end

    s = 0.0
    @inbounds @simd for i in 1:n
        u1 = (z1 - k.x1[i]) / k.h1
        u2 = (z2 - k.x2[i]) / k.h2
        s += _epan2(u1, u2)
    end
    dens = s / (n * k.h1 * k.h2)
    dens = (isfinite(dens) && dens > k.eps_floor) ? dens : k.eps_floor

    lp = log(dens)
    if k.logRT
        lp -= log(v1 + 1)
        lp -= log(v2 + 1)
    end

    return lp
end




#######################################################
############## 2D Joint Gaussian Kernel ###############
#######################################################

#------------- Fit 2D Joint KDE (Gaussian) -------------
function fit_kde2d_gaussian(s1::Vector{Float64}, s2::Vector{Float64}; logRT::Bool=true, bw_rule::Symbol=:silverman, eps_floor::Float64=1e-16)

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

    H = factor^2 .* Σ
    L = cholesky(Symmetric(H)).L

    return KDE2D_Gaussian(X, H, L, logRT, eps_floor)

end


# --------- Evaluate logpdf with Cholesky solve (no explicit inv) ---------
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



#######################################################
############ Mixed SPDF over pairs (c1,c2) ############
#######################################################

function build_mixed2d_spdf(results::AbstractVector, trial::Trial;
                            pairs::Vector{Tuple{Int,Int}},
                            kde_mode::Symbol=:product,            # :product or :gaussian (alias :full)
                            bw_rule::Symbol=:silverman,           # for :gaussian: :scott/:silverman
                            logRT::Bool=true,
                            eps_floor::Float64=1e-16)

    J = length(results)
    if J == 0
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

    if prior <= 1e-2 || J <= 2
        return Mixed2DSPDF(prior, nothing, eps_floor)
    else
        if kde_mode == :product
            kde = fit_kde2d_product(buckets1[p], buckets2[p]; logRT=logRT, eps_floor=eps_floor)
        elseif kde_mode == :gaussian
            kde = fit_kde2d_gaussian(buckets1[p], buckets2[p]; logRT=logRT, bw_rule=bw_rule,eps_floor=eps_floor)
        else
            error("Unknown kde_mode = $kde_mode (use :product or :gaussian).")
        end
        return Mixed2DSPDF(prior, kde, eps_floor)
    end

end

function mixed2d_logpdf(spdf::Mixed2DSPDF, trial::Trial, lambda::Float64=1.0)

    if spdf.kde === nothing
        ll = log(spdf.eps_floor)
    else
        logprior = log(spdf.prior)
        logdens = logpdf(spdf.kde, trial.rt1, trial.rt2)
        ll = logprior + lambda * logdens
    end

    ll = max(ll, log(spdf.eps_floor))

    return ll
end

