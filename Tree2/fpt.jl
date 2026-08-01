using DiffModels

# ----------------- First Passage Time Density -----------------
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

# ----------------- Single trial loglik -----------------
function loglik_trial_stagewise(tr::Trial, φ::Vector{Float64}; eps_floor::Float64=1e-16)
    
    σ = 0.01
    σeff = sqrt(2)*σ
    d1, d2, θ1, θ2, T1, T2  = φ

    r1,r2,r3,r4,r5,r6 = tr.rewards

    # ---- stage1 ----
    μ1   = d1 * (r1 - r2)
    t1 = Float64(tr.rt1) - Float64(T1)

    upper1 = (tr.choice1 == 1)
    g1 = fpt_density_at(t1; μ=μ1, θ=θ1, upper=upper1, σ=σeff)
    # Apply eps_floor to prevent log(0) = -Inf
    g1 = max(g1, eps_floor)

    # ---- stage2 ----
    t2 = Float64(tr.rt2) - Float64(T2)

    if tr.choice1 == 1
        # (LL,LR)
        μ2 = d2 * (r3 - r4)
        upper2 = (tr.choice2 == 1)
    else
        # (RL,RR)
        μ2 = d2 * (r5 - r6)
        upper2 = (tr.choice2 == 3)
    end
    g2 = fpt_density_at(t2; μ=μ2, θ=θ2, upper=upper2, σ=σeff)
    # Apply eps_floor to prevent log(0) = -Inf
    g2 = max(g2, eps_floor)

    return log(g1) + log(g2)
end

# ----------------- First Passage Time CDF -----------------
function fpt_cdf_at(t::Float64; μ::Float64, θ::Float64, upper::Bool, σ::Float64=0.01)
    if t <= 0.0
        return 0.0
    end
    dt = 1.0  # ms
    ts = 0.0:dt:t
    if length(ts) == 0
        return 0.0
    end
    
    pdfs = [fpt_density_at(s; μ=μ, θ=θ, upper=upper, σ=σ) for s in ts if s > 0]
    if isempty(pdfs)
        return 0.0
    end
    
    # Trapezoidal integration
    cdf_val = dt * (sum(pdfs) - 0.5 * (pdfs[1] + pdfs[end]))
    return min(cdf_val, 1.0)
end

# ----------------- Compute analytical CDF for a trial -----------------
function analytical_cdf(tr::Trial, φ::Vector{Float64}; t_values::Vector{Float64})
    σ = 0.01
    σeff = sqrt(2)*σ
    d1, d2, θ1, θ2, T1, T2 = φ
    
    r1, r2, r3, r4, r5, r6 = tr.rewards
    
    # Stage 1
    μ1 = d1 * (r1 - r2)
    upper1 = (tr.choice1 == 1)
    
    # Stage 2
    if tr.choice1 == 1
        μ2 = d2 * (r3 - r4)
        upper2 = (tr.choice2 == 1)
    else
        μ2 = d2 * (r5 - r6)
        upper2 = (tr.choice2 == 3)
    end
    
    cdf_rt1 = Float64[]
    cdf_rt2 = Float64[]
    
    for t in t_values
        # RT1 CDF: time is relative to T1
        t1 = t - T1
        c1 = t1 > 0 ? fpt_cdf_at(t1; μ=μ1, θ=θ1, upper=upper1, σ=σeff) : 0.0
        push!(cdf_rt1, c1)
        
        # RT2 CDF: time is relative to T2
        t2 = t - T2
        c2 = t2 > 0 ? fpt_cdf_at(t2; μ=μ2, θ=θ2, upper=upper2, σ=σeff) : 0.0
        push!(cdf_rt2, c2)
    end
    
    return cdf_rt1, cdf_rt2
end

# ----------------- Compute analytical PDF for a trial -----------------
function analytical_pdf(tr::Trial, φ::Vector{Float64}; t_values::Vector{Float64})
    σ = 0.01
    σeff = sqrt(2)*σ
    d1, d2, θ1, θ2, T1, T2 = φ
    
    r1, r2, r3, r4, r5, r6 = tr.rewards
    
    # Stage 1
    μ1 = d1 * (r1 - r2)
    upper1 = (tr.choice1 == 1)
    
    # Stage 2
    if tr.choice1 == 1
        μ2 = d2 * (r3 - r4)
        upper2 = (tr.choice2 == 1)
    else
        μ2 = d2 * (r5 - r6)
        upper2 = (tr.choice2 == 3)
    end
    
    pdf_rt1 = Float64[]
    pdf_rt2 = Float64[]
    
    for t in t_values
        # RT1 PDF: time is relative to T1
        t1 = t - T1
        p1 = t1 > 0 ? fpt_density_at(t1; μ=μ1, θ=θ1, upper=upper1, σ=σeff) : 0.0
        push!(pdf_rt1, p1)
        
        # RT2 PDF: time is relative to T2
        t2 = t - T2
        p2 = t2 > 0 ? fpt_density_at(t2; μ=μ2, θ=θ2, upper=upper2, σ=σeff) : 0.0
        push!(pdf_rt2, p2)
    end
    
    return pdf_rt1, pdf_rt2
end