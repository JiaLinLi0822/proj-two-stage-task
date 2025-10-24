#!/usr/bin/env julia
# 2D KDE — Silverman rule (native Julia) with robust fallbacks

# 无显示环境下防空白图
ENV["GKSwstype"] = "100"
using Random, LinearAlgebra, Statistics, Distributions, Printf
using Plots; gr()

# ---------------- Silverman 规则 ----------------
silverman_factor(n::Int, d::Int) = (n * (d + 2) / 4)^(-1 / (d + 4))

function silverman_bandwidth_1d(xs::AbstractVector{<:Real}; min_h=1e-6)
    n = length(xs)
    if n <= 1
        return 1.0
    end
    μ  = mean(xs)
    sd = sqrt(mean((xs .- μ).^2))
    q75 = quantile(xs, 0.75); q25 = quantile(xs, 0.25)
    iqr = q75 - q25
    h = 0.9 * min(sd, iqr/1.34) * n^(-1/5)
    if !isfinite(h) || h <= 0
        h = 1.0
    end
    return max(h, min_h)
end

function robust_kde_silverman(X_in::AbstractMatrix{<:Real}; ridge::Float64=1e-8)
    X = Matrix{Float64}(X_in)
    n, d = size(X)
    d == 2 || error("This demo expects 2D data.")
    
    println("="^60)
    println("KDE Analysis for n = $n samples")
    println("="^60)
    
    if n >= 2
        μ = vec(mean(X, dims=1))
        XC = X .- μ'
        Σ  = (XC' * XC) / (n - 1)
        fac = silverman_factor(n, d)
        cov = fac^2 .* Σ

        # Print detailed information
        println("Sample mean: μ = [$(μ[1]), $(μ[2])]")
        println("Sample covariance matrix Σ:")
        for i in 1:size(Σ, 1)
            for j in 1:size(Σ, 2)
                print(@sprintf("%12.6f ", Σ[i, j]))
            end
            println()
        end
        println("Silverman factor: $(@sprintf("%.6f", fac))")
        println("Bandwidth matrix H = factor² × Σ:")
        for i in 1:size(cov, 1)
            for j in 1:size(cov, 2)
                print(@sprintf("%12.6f ", cov[i, j]))
            end
            println()
        end

        L = nothing; ok = false
        for k in 0:6
            λ = ridge * (10.0^k)
            try
                L = cholesky(Symmetric(cov + λ*I)).L
                ok = true
                cov .+= λ*I
                if k > 0
                    println("Ridge regularization applied: λ = $(@sprintf("%.2e", λ))")
                    println("H (after ridge):")
                    for i in 1:size(cov, 1)
                        for j in 1:size(cov, 2)
                            print(@sprintf("%12.6f ", cov[i, j]))
                        end
                        println()
                    end
                end
                break
            catch
            end
        end
        if ok
            log_norm = (d/2)*log(2π) + sum(log.(diag(L)))
            ℓ11, ℓ21, ℓ22 = L[1,1], L[2,1], L[2,2]
            invℓ11, invℓ22 = 1/ℓ11, 1/ℓ22
            evaluator = function (xx::AbstractVector{<:Real}, yy::AbstractVector{<:Real})
                nx, ny = length(xx), length(yy)
                Z = zeros(Float64, ny, nx)
                @inbounds for i in 1:n
                    μ1, μ2 = X[i,1], X[i,2]
                    for (iy, y) in enumerate(yy)
                        u2 = y - μ2
                        for (ix, x) in enumerate(xx)
                            u1 = x - μ1
                            t1 = u1 * invℓ11
                            t2 = (u2 - ℓ21 * t1) * invℓ22
                            e  = 0.5 * (t1*t1 + t2*t2)
                            Z[iy, ix] += exp(-(log_norm + e))
                        end
                    end
                end
                @. Z = max(Z / n, 1e-300)
                Z
            end
            return "silverman.full", Matrix(cov), evaluator
        end
    end
    if n == 0
        println("No data available, using identity bandwidth matrix")
        H = Matrix{Float64}(I, 2, 2)
        eval_empty = (xx, yy) -> fill(1e-300, length(yy), length(xx))
        return "silverman.diag.empty", H, eval_empty
    end
    
    println("Using diagonal bandwidth (fallback for n < 2)")
    h1 = silverman_bandwidth_1d(@view X[:,1])
    h2 = silverman_bandwidth_1d(@view X[:,2])
    println("1D Silverman bandwidths: h1 = $(@sprintf("%.6f", h1)), h2 = $(@sprintf("%.6f", h2))")
    
    H  = Diagonal([h1^2, h2^2]) |> Matrix
    println("Diagonal bandwidth matrix H:")
    for i in 1:size(H, 1)
        for j in 1:size(H, 2)
            print(@sprintf("%12.6f ", H[i, j]))
        end
        println()
    end
    println("="^60)
    
    normconst = 1.0 / (2π*h1*h2)
    evaluator = function (xx::AbstractVector{<:Real}, yy::AbstractVector{<:Real})
        nx, ny = length(xx), length(yy)
        Z = zeros(Float64, ny, nx)
        @inbounds for i in 1:n
            μ1, μ2 = X[i,1], X[i,2]
            for (iy, y) in enumerate(yy)
                dy = (y - μ2)/h2
                for (ix, x) in enumerate(xx)
                    dx = (x - μ1)/h1
                    Z[iy, ix] += exp(-0.5*(dx*dx + dy*dy))
                end
            end
        end
        @. Z = max(normconst * (Z / n), 1e-300)
        Z
    end
    return "silverman.diag", H, evaluator
end

function main()
    Random.seed!(42)
    μ = [0.0, 0.0]
    Σ = [1.0 0.7; 0.7 1.5]
    dist = MvNormal(μ, Σ)

    n_list = [1, 2, 3, 5, 10, 30, 100, 200, 500]
    k  = length(n_list)
    nc = ceil(Int, sqrt(k))
    nr = ceil(Int, k/nc)

    default(fmt = :png, size=(360*nc, 320*nr))
    plt = plot(layout = (nr, nc), legend = false)

    results = Vector{Tuple{Int,String,Matrix{Float64}}}()

    for (i, n) in enumerate(n_list)
        X = permutedims(rand(dist, n))
        method, H, evaluator = robust_kde_silverman(X)

        stds = sqrt.(diag(Σ))
        x1_min, x1_max = μ[1] - 3*stds[1], μ[1] + 3*stds[1]
        x2_min, x2_max = μ[2] - 3*stds[2], μ[2] + 3*stds[2]
        if n > 0
            x1_min = min(x1_min, minimum(view(X,:,1)) - 1.0)
            x1_max = max(x1_max, maximum(view(X,:,1)) + 1.0)
            x2_min = min(x2_min, minimum(view(X,:,2)) - 1.0)
            x2_max = max(x2_max, maximum(view(X,:,2)) + 1.0)
        end

        xx = collect(range(x1_min, x1_max; length=200))
        yy = collect(range(x2_min, x2_max; length=200))
        Z  = evaluator(xx, yy)

        zmin, zmax = extrema(Z)

        # 直接往总画布的第 i 个子图上画
        heatmap!(plt, xx, yy, Z;
                 subplot = i,
                 colorbar = false,
                 c = :viridis,
                 clims = (zmin, zmax),
                 xlims = (x1_min, x1_max),
                 ylims = (x2_min, x2_max),
                 aspect_ratio = :equal,
                 framestyle = :box,
                 dpi = 200)

        contour!(plt, xx, yy, Z;
                 subplot = i,
                 levels = quantile(vec(Z), [0.25, 0.5, 0.75, 0.9, 0.95]),
                 linewidth = 0.8)

        if n > 0
            scatter!(plt, view(X,:,1), view(X,:,2);
                     subplot = i,
                     ms = 2.5, mc = :black, ma = 0.7)
        end

        # 记录 H
        push!(results, (n, method, H))
    end

    savefig(plt, "kde_silverman_demo.png")
    println("Saved figure to kde_silverman_demo.png")

    println("\n===== Bandwidth matrices (H) by sample size (Silverman/native) =====")
    for (n, method, H) in results
        println("n = $(lpad(n,3)) | method = $(lpad(method,20)) | H =")
        show(stdout, "text/plain", round.(H, digits=6)); println("\n")
    end
end

main()