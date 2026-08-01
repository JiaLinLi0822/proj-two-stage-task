#!/usr/bin/env julia

using DiffModels
using Random
using Plots
using Statistics

function fpt_density_at(t::Float64; μ::Float64, θ::Float64, upper::Bool, σ::Float64=0.01)
    if t <= 0.0
        return 0.0
    end

    μs = μ/σ
    θs = θ/σ

    dt  = 1
    d   = ConstDrift(μs, dt)
    Bs  = ConstSymBounds(θs, dt)
    
    if upper
        return pdfu(d, Bs, t)
    else
        return pdfl(d, Bs, t)
    end
end

function compute_distributions(μ::Float64, θ::Float64, σ::Float64; tmax::Float64=10000.0, dt::Float64=1.0)
    """
    Compute PDF and CDF for both upper and lower boundaries.
    
    Returns:
        ts: time points
        pdf_upper: PDF for upper boundary
        pdf_lower: PDF for lower boundary
        cdf_upper: CDF for upper boundary (integral of PDF)
        cdf_lower: CDF for lower boundary (integral of PDF)
    """
    ts = collect(0.0:dt:tmax)
    
    pdf_upper = [fpt_density_at(t; μ=μ, θ=θ, upper=true, σ=σ) for t in ts]
    pdf_lower = [fpt_density_at(t; μ=μ, θ=θ, upper=false, σ=σ) for t in ts]
    
    cdf_upper = zeros(length(ts))
    cdf_lower = zeros(length(ts))

    cumsum_upper = cumsum(pdf_upper)
    cumsum_lower = cumsum(pdf_lower)
    
    for i in 1:length(ts)
        if i == 1
            cdf_upper[i] = dt * pdf_upper[i]
            cdf_lower[i] = dt * pdf_lower[i]
        else
            # Trapezoidal integration from 0 to ts[i]
            cdf_upper[i] = dt * (cumsum_upper[i] - 0.5 * (pdf_upper[1] + pdf_upper[i]))
            cdf_lower[i] = dt * (cumsum_lower[i] - 0.5 * (pdf_lower[1] + pdf_lower[i]))
        end
    end
    
    return ts, pdf_upper, pdf_lower, cdf_upper, cdf_lower
end

function main()
    Random.seed!(42)

    μ = 4e-5
    θ = 0.3
    σ = 0.01
    
    ts, pdf_upper, pdf_lower, cdf_upper, cdf_lower = compute_distributions(μ, θ, σ)
    
    total_prob_upper = sum(pdf_upper) * 1.0
    total_prob_lower = sum(pdf_lower) * 1.0
    
    println("Total probabilities (integrals):")
    println("  Upper boundary: $(round(total_prob_upper, digits=4))")
    println("  Lower boundary: $(round(total_prob_lower, digits=4))")
    println("  Total: $(round(total_prob_upper + total_prob_lower, digits=4))")
    println()
    
    # Plot 1: PDFs
    p1 = plot(ts, pdf_upper, 
              label="Upper boundary", 
              linewidth=2, 
              color=:blue,
              xlabel="Time (ms)", 
              ylabel="PDF",
              legend=:topright)
    
    plot!(p1, ts, pdf_lower, 
          label="Lower boundary", 
          linewidth=2, 
          color=:red)
    
    # Plot 2: CDFs
    p2 = plot(ts, cdf_upper, 
              label="Upper boundary", 
              linewidth=2, 
              color=:blue,
              xlabel="Time (ms)", 
              ylabel="CDF",
              legend=:bottomright)
    
    plot!(p2, ts, cdf_lower, 
          label="Lower boundary", 
          linewidth=2, 
          color=:red)
    
    p_combined = plot(p1, p2, layout=(2, 1), size=(800, 600))
    
    output_file = "Tree2/figures/fpt_example.png"
    mkpath(dirname(output_file))
    savefig(p_combined, output_file)
    
    println("Plot saved to: $output_file")
    
    display(p_combined)
    
    return ts, pdf_upper, pdf_lower, cdf_upper, cdf_lower
end

main()