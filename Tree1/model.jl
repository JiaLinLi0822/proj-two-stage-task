using Random
include("data.jl")

struct Model
    simulate::Function
    θ::Vector{Float64} # parameters
end

function simulate(m::Model, trial::Trial)
    m.simulate(m.θ, trial.rewards)
end


# ======== Primary two-stage independent paths model ========
"""
Two-stage sequential integration model with independent paths.

Parameters:
- θ[1] (d1): Drift rate for first stage evidence accumulation
- θ[2] (d2): Drift rate for second stage evidence accumulation  
- θ[3] (θ1): Decision threshold for first stage (difference between top 2 paths)
- θ[4] (θ2): Decision threshold for second stage (difference between remaining paths)
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- First stage: Accumulates evidence for all 4 paths (LL, LR, RL, RR) in parallel
- Decision: Made when difference between top 2 paths exceeds θ1
- Second stage: Only considers paths from chosen side (L or R)
- Final choice: Determined by argmax of remaining evidence
"""
function model1(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d1, d2      = θ[1], θ[2]
    θ1, θ2      = θ[3], θ[4]
    T1, T2      = θ[5], θ[6]
    sigma       = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t, rt1, choice1 = 0, 0, 0
    max_step = 10000
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break 
        end
        # update evidence for each path
        E[1] += (d1 * rewards[1]  + randn()*sigma) + (d1 * rewards[3] + randn()*sigma)
        E[2] += (d1 * rewards[2]  + randn()*sigma) + (d1 * rewards[4] + randn()*sigma)
        E[3] += (d1 * rewards[2]  + randn()*sigma) + (d1 * rewards[5] + randn()*sigma)

        # check if top‐2 difference reaches θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ1
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- second stage ---
    t2 = 0
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        rt2 = T2
        choice2 = 1
    else
        E2 = copy(E[2:3])
        while true
            t2 += 1
            
            if t2 == max_step
                timeout = true
                break
            end
            
            E2[1] += (d2 * rewards[4] + randn()*sigma)
            E2[2] += (d2 * rewards[5] + randn()*sigma)

            sorted_E2 = sort(E2, rev=true)
            if abs(sorted_E2[1] - sorted_E2[2]) ≥ θ2
                rt2    = t2 + T2
                idx2 = argmax(E2)
                choice2 = idx2 == 1 ? 2 : 3
                break
            end
        end
    end
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end




# ======== Two stages with correlated paths model ========
"""
Two-stage sequential integration model with correlated noise between paths.

Parameters:
- θ[1] (d1): Drift rate for first stage evidence accumulation
- θ[2] (d2): Drift rate for second stage evidence accumulation  
- θ[3] (θ1): Decision threshold for first stage (difference between top 2 paths)
- θ[4] (θ2): Decision threshold for second stage (difference between remaining paths)
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- First stage: Accumulates evidence with correlated noise between L/R branches
- Correlation: Same noise for paths sharing first-stage choice (L or R)
- Decision: Made when difference between top 2 paths exceeds θ1
- Second stage: Only considers paths from chosen side (L or R)
- Final choice: Determined by argmax of remaining evidence
"""
function model2(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d1, d2      = θ[1], θ[2]
    θ1, θ2      = θ[3], θ[4]
    T1, T2      = θ[5], θ[6]
    sigma       = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t, rt1, choice1 = 0, 0, 0
    max_step = 10000
    
    # --- first stage ---
    while true
        t += 1
        if t >= max_step
            timeout = true
            break
        end

        # Correlated noise for R_L and R_R
        noise_L = randn() * sigma
        noise_R = randn() * sigma

        # update evidence for each path
        E[1] += (d1 * rewards[1]  + noise_L) + (d1 * rewards[3] + randn()*sigma)
        E[2] += (d1 * rewards[2]  + noise_R) + (d1 * rewards[4] + randn()*sigma)
        E[3] += (d1 * rewards[2]  + noise_R) + (d1 * rewards[5] + randn()*sigma)

        # check if top‐2 difference reaches θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ1
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- second stage ---
    t2 = 0
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        rt2 = T2
        choice2 = 1
    else
        E2 = copy(E[2:3])
        while true
            t2 += 1
            
            if t2 == max_step
                timeout = true
                break
            end
            
            E2[1] += (d2 * rewards[4] + randn()*sigma)
            E2[2] += (d2 * rewards[5] + randn()*sigma)

            sorted_E2 = sort(E2, rev=true)
            if abs(sorted_E2[1] - sorted_E2[2]) ≥ θ2
                rt2    = t2 + T2
                idx2 = argmax(E2)
                choice2 = idx2 == 1 ? 2 : 3
                break
            end
        end
    end
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end




# ======== Two stages, independent path with pruning model ========
"""
Two-stage sequential integration model with independent paths and pruning mechanism.

Parameters:
- θ[1] (d1): Drift rate for first stage evidence accumulation
- θ[2] (d2): Drift rate for second stage evidence accumulation  
- θ[3] (θ1): Decision threshold for first stage (difference between top 2 paths)
- θ[4] (θ2): Decision threshold for second stage (difference between remaining paths)
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)
- θ[7] (θ_prun): Pruning threshold (difference between L and R side evidence)

Model behavior:
- First stage: Accumulates evidence for all 4 paths (LL, LR, RL, RR) in parallel
- Pruning: Early termination if one side clearly dominates the other (exceeds θ_prun)
- Decision: Made when difference between top 2 paths exceeds θ1 OR pruning occurs
- Second stage: Only considers paths from chosen side (L or R)
- Final choice: Determined by argmax of remaining evidence
"""
function model3(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d1, d2      = θ[1], θ[2]
    θ1, θ2      = θ[3], θ[4]
    T1, T2      = θ[5], θ[6]
    θ_prun      = θ[7]
    sigma       = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t, rt1, choice1 = 0, 0, 0
    max_step = 10000
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end
        # update evidence for each path
        E[1] += (d1 * rewards[1]  + randn()*sigma) + (d1 * rewards[3] + randn()*sigma)
        E[2] += (d1 * rewards[2]  + randn()*sigma) + (d1 * rewards[4] + randn()*sigma)
        E[3] += (d1 * rewards[2]  + randn()*sigma) + (d1 * rewards[5] + randn()*sigma)

        # check if top‐2 difference reaches θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ1
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end

        # pruning
        if min(E[1]) - max(E[2], E[3]) ≥ θ_prun || min(E[2], E[3]) - max(E[1]) ≥ θ_prun
            rt1 = t + T1
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    # --- second stage ---
    t2 = 0
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        rt2 = T2
        choice2 = 1
    else
        E2 = copy(E[2:3])
        while true
            t2 += 1
            
            if t2 == max_step
                timeout = true
                break
            end
            
            E2[1] += (d2 * rewards[4] + randn()*sigma)
            E2[2] += (d2 * rewards[5] + randn()*sigma)

            sorted_E2 = sort(E2, rev=true)
            if abs(sorted_E2[1] - sorted_E2[2]) ≥ θ2
                rt2    = t2 + T2
                idx2 = argmax(E2)
                choice2 = idx2 == 1 ? 2 : 3
                break
            end
        end
    end
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== Two stages, correlated path with pruning model ========
"""
Two-stage sequential integration model with correlated noise and pruning mechanism.

Parameters:
- θ[1] (d1): Drift rate for first stage evidence accumulation
- θ[2] (d2): Drift rate for second stage evidence accumulation  
- θ[3] (θ1): Decision threshold for first stage (difference between top 2 paths)
- θ[4] (θ2): Decision threshold for second stage (difference between remaining paths)
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)
- θ[7] (θ_prun): Pruning threshold (difference between L and R side evidence)

Model behavior:
- First stage: Accumulates evidence with correlated noise between L/R branches
- Correlation: Same noise for paths sharing first-stage choice (L or R)
- Pruning: Early termination if one side clearly dominates the other (exceeds θ_prun)
- Decision: Made when difference between top 2 paths exceeds θ1 OR pruning occurs
- Second stage: Only considers paths from chosen side (L or R)
- Final choice: Determined by argmax of remaining evidence
"""
function model4(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d1, d2      = θ[1], θ[2]
    θ1, θ2      = θ[3], θ[4]
    T1, T2      = θ[5], θ[6]
    θ_prun      = θ[7]
    sigma       = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t, rt1, choice1 = 0, 0, 0
    max_step = 10000
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end
    
        # Correlated noise for R_L and R_R
        noise_L = randn() * sigma
        noise_R = randn() * sigma

        # update evidence for each path
        E[1] += (d1 * rewards[1]  + randn()*sigma) + (d1 * rewards[3] + randn()*sigma)
        E[2] += (d1 * rewards[2]  + randn()*sigma) + (d1 * rewards[4] + randn()*sigma)
        E[3] += (d1 * rewards[2]  + randn()*sigma) + (d1 * rewards[5] + randn()*sigma)

        # check if top‐2 difference reaches θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ1
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end

        # pruning
        if min(E[1]) - max(E[2], E[3]) ≥ θ_prun || min(E[2], E[3]) - max(E[1]) ≥ θ_prun
            rt1 = t + T1
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- second stage ---
    t2 = 0
    rt2 = 0.0

    # --- second stage ---
    t2 = 0
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        rt2 = T2
        choice2 = 1
    else
        E2 = copy(E[2:3])
        while true
            t2 += 1
            
            if t2 == max_step
                timeout = true
                break
            end
            
            E2[1] += (d2 * rewards[4] + randn()*sigma)
            E2[2] += (d2 * rewards[5] + randn()*sigma)

            sorted_E2 = sort(E2, rev=true)
            if abs(sorted_E2[1] - sorted_E2[2]) ≥ θ2
                rt2    = t2 + T2
                idx2 = argmax(E2)
                choice2 = idx2 == 1 ? 2 : 3
                break
            end
        end
    end
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== Two-stage, independent paths, single drift ========
"""
Two-stage sequential integration model with independent paths and single drift rate.

Parameters:
- θ[1] (d): Drift rate for both stages evidence accumulation (shared parameter)
- θ[2] (θ1): Decision threshold for first stage (difference between top 2 paths)
- θ[3] (θ2): Decision threshold for second stage (difference between remaining paths)
- θ[4] (T1): Non-decision time for first stage (baseline response time)
- θ[5] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- First stage: Accumulates evidence for all 4 paths (LL, LR, RL, RR) in parallel
- Decision: Made when difference between top 2 paths exceeds θ1
- Second stage: Only considers paths from chosen side (L or R)
- Final choice: Determined by argmax of remaining evidence
- Note: Uses same drift rate (d) for both stages, reducing parameter count
"""
function model5(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d           = θ[1]
    θ1, θ2      = θ[2], θ[3]
    T1, T2      = θ[4], θ[5]
    sigma       = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t, rt1, choice1 = 0, 0, 0
    max_step = 10000
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end
        # update evidence for each path
        E[1] += (d * rewards[1]  + randn()*sigma) + (d * rewards[3] + randn()*sigma)
        E[2] += (d * rewards[2]  + randn()*sigma) + (d * rewards[4] + randn()*sigma)
        E[3] += (d * rewards[2]  + randn()*sigma) + (d * rewards[5] + randn()*sigma)

        # check if top‐2 difference reaches θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ1
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- second stage ---
    t2 = 0
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        rt2 = T2
        choice2 = 1
    else
        E2 = copy(E[2:3])
        while true
            t2 += 1
            
            if t2 == max_step
                timeout = true
                break
            end
            
            E2[1] += (d2 * rewards[4] + randn()*sigma)
            E2[2] += (d2 * rewards[5] + randn()*sigma)

            sorted_E2 = sort(E2, rev=true)
            if abs(sorted_E2[1] - sorted_E2[2]) ≥ θ2
                rt2    = t2 + T2
                idx2 = argmax(E2)
                choice2 = idx2 == 1 ? 2 : 3
                break
            end
        end
    end
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== Forward greedy search ========
"""
Forward greedy search model with sequential decision making.

Parameters:
- θ[1] (d1): Drift rate for first stage evidence accumulation (L vs R)
- θ[2] (d2): Drift rate for second stage evidence accumulation (within chosen side)
- θ[3] (θ1): Decision threshold for first stage (difference between L and R)
- θ[4] (θ2): Decision threshold for second stage (difference between remaining options)
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- First stage: Accumulates evidence for L vs R choice only
- Decision: Made when difference between L and R exceeds θ1
- Second stage: Only considers options within chosen side (LL/LR or RL/RR)
- Final choice: Determined by argmax within chosen side
- Note: Greedy approach - makes first decision without considering full tree
"""
function model6(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d1, d2      = θ[1], θ[2]
    θ1, θ2      = θ[3], θ[4]
    T1, T2      = θ[5], θ[6]
    sigma       = 0.01

    # initialize evidence for four paths
    E1 = zeros(2) # [L, R]
    timeout = false

    # --- first stage ---
    t = 0
    rt1 = 0.0
    max_step = 10000
    choice1 = 0
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end

        # accumulate into first stage integrators
        E1[1] += (d1 * rewards[1]  + randn()*sigma)
        E1[2] += (d1 * rewards[2]  + randn()*sigma)

        # check first-stage threshold
        if abs(E1[1] - E1[2]) >= θ1
            rt1    = t + T1
            choice1 = E1[1] > E1[2] ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # ---- Second stage ----
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        # Left chosen → only LL exists (forced)
        rt2     = T2
        choice2 = 1    # path index 1 = LL
    else
        # Right chosen → need to accumulate RL vs RR
        E2 = zeros(2)   # [RL, RR]
        t2 = 0

        while true
            t2 += 1
            if t2 == max_step
                timeout = true
                break
            end

            # updates for RL and RR
            E2[1] += d2 * rewards[4] + randn()*sigma   # RL
            E2[2] += d2 * rewards[5] + randn()*sigma   # RR

            if abs(E2[1] - E2[2]) ≥ θ2
                rt2 = t2 + T2
                # map back to global path index: RL->2, RR->3
                choice2 = E2[1] > E2[2] ? 2 : 3
                break
            end
        end
    end
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== Backward search ========
"""
Backward search model with bottom-up evaluation strategy.

Parameters:
- θ[1] (d0): Drift rate for leaf-level evidence accumulation (stage 0)
- θ[2] (d1): Drift rate for top-level evidence accumulation (stage 1)
- θ[3] (d2): Drift rate for final choice evidence accumulation (stage 2)
- θ[4] (θ0): Decision threshold for leaf-level decisions
- θ[5] (θ1): Decision threshold for top-level decisions
- θ[6] (θ2): Decision threshold for final choice decisions
- θ[7] (T1): Non-decision time for first stage (baseline response time)
- θ[8] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- Stage 0: Evaluates leaf nodes (LL vs LR, RL vs RR) in parallel
- Stage 1: Uses frozen leaf values to make top-level decision (L vs R)
- Stage 2: Unfreezes chosen branch for final choice within that side
- Note: Backward approach - evaluates leaves first, then works up the tree
"""
function model7(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d0, d1, d2  = θ[1], θ[2], θ[3]
    θ0, θ1, θ2  = θ[4], θ[5], θ[6]
    T1, T2      = θ[7], θ[8]
    sigma       = 0.01

    timeout = false
    max_step = 10000

    # ---- stage 0 ------#
    # Left‐branch integrators
    E_LL = 0.0
    tL = 0
    while abs(E_LL) < θ0
        tL += 1
        if tL == max_step
            timeout = true
            break
        end

        E_LL += d0 * rewards[3] + randn()*sigma
    end
    # Right‐branch integrators
    E_RL = 0.0; E_RR = 0.0
    tR = 0
    while abs(E_RL - E_RR) < θ0
        tR += 1
        if tR == max_step
            timeout = true
            break
        end

        E_RL += d0 * rewards[4] + randn()*sigma
        E_RR += d0 * rewards[5] + randn()*sigma
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # Determine winning leaf and its value
    win_left_val  = E_LL
    win_right_val = max(E_RL, E_RR)

    rt0 = max(tL, tR)

    # --- Stage 1: Top‐level competition (Eq. S8–S9)  ---
    E_top_L = 0.0; E_top_R = 0.0
    t1 = 0
    rt1 = 0
    choice1 = 0
    while true
        t1 += 1

        if t1 == max_step
            timeout = true
            break
        end

        E_top_L += d1 * rewards[1] + randn()*sigma
        E_top_R += d1 * rewards[2] + randn()*sigma

        # internal decision variable includes the frozen leaf‐value
        Δ = (E_top_L + win_left_val) - (E_top_R + win_right_val)
        if abs(Δ) ≥ θ1
            rt1 = rt0 + t1 + T1
            choice1 = Δ > 0 ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- Stage 2: Unfreeze chosen branch (only for right side) ---
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        # forced LL
        rt2 = T2
        choice2 = 1
    else
        # continue integration for RL vs RR until θ2
        v1, v2 = E_RL, E_RR
        t2 = 0
        while abs(v1 - v2) < θ2
            t2 += 1
            if t2 == max_step
                timeout = true
                break
            end
            v1 += d2 * rewards[4] + randn()*sigma  # RL
            v2 += d2 * rewards[5] + randn()*sigma  # RR
        end
        rt2 = t2 + T2
        choice2 = v1 > v2 ? 2 : 3
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== Backward search , same first-stage parameters========
"""
Backward search model with shared parameters for leaf and top-level decisions.

Parameters:
- θ[1] (d0): Drift rate for both leaf-level and top-level evidence accumulation
- θ[2] (d2): Drift rate for final choice evidence accumulation (stage 2)
- θ[3] (θ0): Decision threshold for both leaf-level and top-level decisions
- θ[4] (θ2): Decision threshold for final choice decisions
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- Stage 0: Evaluates leaf nodes (LL vs LR, RL vs RR) in parallel
- Stage 1: Uses frozen leaf values to make top-level decision (L vs R)
- Stage 2: Unfreezes chosen branch for final choice within that side
- Note: Uses same drift rate and threshold for stages 0 and 1, reducing parameters
"""
function model8(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d0, d2      = θ[1], θ[2]
    θ0, θ2      = θ[3], θ[4]
    T1, T2      = θ[5], θ[6]
    sigma       = 0.01

    timeout = false
    max_step = 10000

    # ---- stage 0 ------#
    # Left‐branch integrators
    E_LL = 0.0
    tL = 0
    while abs(E_LL) < θ0
        tL += 1
        if tL == max_step
            timeout = true
            break
        end

        E_LL += d0 * rewards[3] + randn()*sigma
    end
    # Right‐branch integrators
    E_RL = 0.0; E_RR = 0.0
    tR = 0
    while abs(E_RL - E_RR) < θ0
        tR += 1
        if tR == max_step
            timeout = true
            break
        end

        E_RL += d0 * rewards[4] + randn()*sigma
        E_RR += d0 * rewards[5] + randn()*sigma
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # Determine winning leaf and its value
    win_left_val  = E_LL
    win_right_val = max(E_RL, E_RR)

    rt0 = max(tL, tR)

    # --- Stage 1: Top‐level competition (Eq. S8–S9)  ---
    E_top_L = 0.0; E_top_R = 0.0
    t1 = 0
    rt1 = 0
    choice1 = 0
    while true
        t1 += 1

        if t1 == max_step
            timeout = true
            break
        end

        E_top_L += d0 * rewards[1] + randn()*sigma
        E_top_R += d0 * rewards[2] + randn()*sigma

        # internal decision variable includes the frozen leaf‐value
        Δ = (E_top_L + win_left_val) - (E_top_R + win_right_val)
        if abs(Δ) ≥ θ0
            rt1 = rt0 + t1 + T1
            choice1 = Δ > 0 ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- Stage 2: Unfreeze chosen branch (additive DDM, Eq. S2) ---
    # Select the two leaf integrators for the chosen side
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        # forced LL
        rt2 = T2
        choice2 = 1
    else
        # continue integration for RL vs RR until θ2
        v1, v2 = E_RL, E_RR
        t2 = 0
        while abs(v1 - v2) < θ2
            t2 += 1
            if t2 == max_step
                timeout = true
                break
            end
            v1 += d2 * rewards[4] + randn()*sigma  # RL
            v2 += d2 * rewards[5] + randn()*sigma  # RR
        end
        rt2 = t2 + T2
        choice2 = v1 > v2 ? 2 : 3
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end






# ======== Backward search with reset ========
"""
Backward search model with reset mechanism for evidence accumulation.

Parameters:
- θ[1] (d0): Drift rate for leaf-level evidence accumulation (stage 0)
- θ[2] (d1): Drift rate for top-level evidence accumulation (stage 1)
- θ[3] (d2): Drift rate for final choice evidence accumulation (stage 2)
- θ[4] (θ0): Decision threshold for leaf-level decisions
- θ[5] (θ1): Decision threshold for top-level decisions
- θ[6] (θ2): Decision threshold for final choice decisions
- θ[7] (T1): Non-decision time for first stage (baseline response time)
- θ[8] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- Stage 0: Evaluates leaf nodes (LL vs LR, RL vs RR) in parallel
- Stage 1: Resets evidence and uses frozen leaf values for top-level decision
- Stage 2: Unfreezes chosen branch for final choice within that side
- Note: Reset mechanism - evidence accumulation starts fresh in stage 1
"""
function model9(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d0, d1, d2  = θ[1], θ[2], θ[3]
    θ0, θ1, θ2  = θ[4], θ[5], θ[6]
    T1, T2      = θ[7], θ[8]
    sigma       = 0.01

    timeout = false
    max_step = 10000

    # ---- stage 0 ------#
    # Left‐branch integrators
    E_LL = 0.0
    tL = 0
    while abs(E_LL) < θ0
        tL += 1
        if tL == max_step
            timeout = true
            break
        end

        E_LL += d0 * rewards[3] + randn()*sigma
    end
    # Right‐branch integrators
    E_RL = 0.0; E_RR = 0.0
    tR = 0
    while abs(E_RL - E_RR) < θ0
        tR += 1
        if tR == max_step
            timeout = true
            break
        end

        E_RL += d0 * rewards[4] + randn()*sigma
        E_RR += d0 * rewards[5] + randn()*sigma
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # Determine winning leaf and its value
    winL = 3 # left winner is always LL
    winR = (E_RL > E_RR) ? 4 : 5

    rt0 = max(tL, tR)

    # ── STAGE 1 WITH RESET ──
    # prune away losers and start both survivors at zero
    E_top_L = 0.0
    E_top_R = 0.0
    t1      = 0
    choice1 = 0
    rt1 = 0

    while true
        t1 += 1
        if t1 >= max_step
            timeout = true
            break
        end

        # accumulate trunk + winning‐leaf in parallel
        E_top_L += (d1*rewards[1] + randn()*sigma) + (d1*rewards[winL] + randn()*sigma)
        E_top_R += (d1*rewards[2] + randn()*sigma) + (d1*rewards[winR] + randn()*sigma)

        # now compare the *reset* survivors
        if abs(E_top_L - E_top_R) ≥ θ1
            rt1     = rt0 + t1 + T1
            choice1 = (E_top_L > E_top_R) ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- Stage 2: Unfreeze chosen branch (additive DDM, Eq. S2) ---
    # Select the two leaf integrators for the chosen side
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        # left chosen: forced LL
        rt2     = T2
        choice2 = 1
    else
        # right chosen: continue RL vs RR
        v1, v2 = E_RL, E_RR
        t2     = 0
        while abs(v1 - v2) < θ2
            t2 += 1
            if t2 == max_step
                timeout = true; break
            end
            v1 += d2 * rewards[4] + randn()*sigma
            v2 += d2 * rewards[5] + randn()*sigma
        end
        rt2     = t2 + T2
        choice2 = (v1 > v2) ? 2 : 3
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end






# ======== Backward search with reset, same first-stage parameters ========
"""
Backward search model with reset mechanism and shared parameters.

Parameters:
- θ[1] (d0): Drift rate for both leaf-level and top-level evidence accumulation
- θ[2] (d2): Drift rate for final choice evidence accumulation (stage 2)
- θ[3] (θ0): Decision threshold for both leaf-level and top-level decisions
- θ[4] (θ2): Decision threshold for final choice decisions
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- Stage 0: Evaluates leaf nodes (LL vs LR, RL vs RR) in parallel
- Stage 1: Resets evidence and uses frozen leaf values for top-level decision
- Stage 2: Unfreezes chosen branch for final choice within that side
- Note: Uses same drift rate and threshold for stages 0 and 1, with reset mechanism
"""
function model10(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d0, d2  = θ[1], θ[2]
    θ0, θ2  = θ[3], θ[4]
    T1, T2  = θ[5], θ[6]
    sigma       = 0.01

    timeout = false
    max_step = 10000

    # ---- stage 0 ------#
    # Left‐branch integrators
    E_LL = 0.0
    tL   = 0
    while abs(E_LL) < θ0
        tL += 1
        if tL == max_step
            timeout = true; break
        end
        E_LL += d0 * rewards[3] + randn()*sigma
    end
    # Right‐branch integrators
    E_RL = 0.0; E_RR = 0.0
    tR   = 0
    while abs(E_RL - E_RR) < θ0
        tR += 1
        if tR == max_step
            timeout = true; break
        end
        E_RL += d0 * rewards[4] + randn()*sigma
        E_RR += d0 * rewards[5] + randn()*sigma
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # Determine winning leaf and its value
    winL = 3 # left winner is always LL
    winR = (E_RL > E_RR) ? 4 : 5

    rt0 = max(tL, tR)

    # ── STAGE 1 WITH RESET ──
    # prune away losers and start both survivors at zero
    E_top_L = 0.0
    E_top_R = 0.0
    t1      = 0
    choice1 = 0
    rt1 = 0

    while true
        t1 += 1
        if t1 >= max_step
            timeout = true
            break
        end

        # accumulate trunk + winning‐leaf in parallel
        E_top_L += (d0*rewards[1] + randn()*sigma) + (d0*rewards[winL] + randn()*sigma)
        E_top_R += (d0*rewards[2] + randn()*sigma) + (d0*rewards[winR] + randn()*sigma)

        # now compare the *reset* survivors
        if abs(E_top_L - E_top_R) ≥ θ0
            rt1     = rt0 + t1 + T1
            choice1 = (E_top_L > E_top_R) ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- Stage 2: Unfreeze chosen branch (additive DDM, Eq. S2) ---
    # Select the two leaf integrators for the chosen side
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        # left chosen: forced LL
        rt2     = T2
        choice2 = 1
    else
        # right chosen: continue RL vs RR
        v1, v2 = E_RL, E_RR
        t2     = 0
        while abs(v1 - v2) < θ2
            t2 += 1
            if t2 == max_step
                timeout = true; break
            end
            v1 += d2 * rewards[4] + randn()*sigma
            v2 += d2 * rewards[5] + randn()*sigma
        end
        rt2     = t2 + T2
        choice2 = (v1 > v2) ? 2 : 3
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== One-stage parallel integration with vigor model ========
"""
One-stage parallel integration model with vigor-based response time adjustment.

Parameters:
- θ[1] (d): Drift rate for evidence accumulation across all paths
- θ[2] (θ): Decision threshold (difference between top 2 paths)
- θ[3] (vigor1): Vigor parameter for first stage response time adjustment
- θ[4] (vigor2): Vigor parameter for second stage response time adjustment
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- Single stage: Accumulates evidence for all 4 paths (LL, LR, RL, RR) in parallel
- Decision: Made when difference between top 2 paths exceeds θ
- Vigor effect: Response time adjusted based on difference from maximum reward
- RT1 = T1 + vigor1 * (rmax - r1w)
- RT2 = T2 + vigor2 * (rmax - r2w)
- Note: One-stage model with vigor-based response time modulation
"""
function model11(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d              = θ[1]
    θ0             = θ[2]
    vigor1, vigor2 = θ[3], θ[4]
    T1, T2         = θ[5], θ[6]
    sigma          = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t, rt1, choice1 = 0, 0.0, 0
    max_step = 10000
    idx = 0
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end
        # update evidence for each path
        E[1] += (d * rewards[1]  + randn()*sigma) + (d * rewards[3] + randn()*sigma)
        E[2] += (d * rewards[2]  + randn()*sigma) + (d * rewards[4] + randn()*sigma)
        E[3] += (d * rewards[2]  + randn()*sigma) + (d * rewards[5] + randn()*sigma)

        # check if the leading difference exceeds threshold θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ0
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    # --- second stage ---
    rt2, choice2 = 0.0, 0

    if idx == 1  # LL path
        r1w = rewards[1]
        r2w = rewards[3]
        choice2 = 1
    elseif idx == 2  # RL path
        r1w = rewards[2]
        r2w = rewards[4]
        choice2 = 2
    else  # RR path
        r1w = rewards[2]
        r2w = rewards[5]
        choice2 = 3
    end

    rmax = maximum(rewards)
    rt1 += T1 + vigor1 * (rmax - r1w)
    rt2 += T2 + vigor2 * (rmax - r2w)
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end




# ======== One-stage parallel integration with single vigor model ========
"""
One-stage parallel integration model with single vigor parameter.

Parameters:
- θ[1] (d): Drift rate for evidence accumulation across all paths
- θ[2] (θ): Decision threshold (difference between top 2 paths)
- θ[3] (vigor): Single vigor parameter for response time adjustment
- θ[4] (T1): Non-decision time for first stage (baseline response time)
- θ[5] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- Single stage: Accumulates evidence for all 4 paths (LL, LR, RL, RR) in parallel
- Decision: Made when difference between top 2 paths exceeds θ
- Vigor effect: Response time adjusted based on difference from maximum reward
- RT1 = T1 + vigor * (rmax - r1w)
- RT2 = T2 + vigor * (rmax - r2w)
- Note: Uses single vigor parameter for both stages, reducing parameter count
"""
function model12(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d              = θ[1]
    θ0             = θ[2]
    vigor          = θ[3]
    T1, T2         = θ[4], θ[5]
    sigma          = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t, rt1, choice1 = 0, 0.0, 0
    max_step = 10000
    idx = 0
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end
        # update evidence for each path
        E[1] += (d * rewards[1]  + randn()*sigma) + (d * rewards[3] + randn()*sigma)
        E[2] += (d * rewards[2]  + randn()*sigma) + (d * rewards[4] + randn()*sigma)
        E[3] += (d * rewards[2]  + randn()*sigma) + (d * rewards[5] + randn()*sigma)

        # check if the leading difference exceeds threshold θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ0
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    # --- second stage ---
    rt2, choice2 = 0.0, 0

    if idx == 1  # LL path
        r1w = rewards[1]
        r2w = rewards[3]
        choice2 = 1
    elseif idx == 2  # RL path
        r1w = rewards[2]
        r2w = rewards[4]
        choice2 = 2
    else  # RR path
        r1w = rewards[2]
        r2w = rewards[5]
        choice2 = 3
    end

    rmax = maximum(rewards)
    rt1 += T1 + vigor * (rmax - r1w)
    rt2 += T2 + vigor * (rmax - r2w)
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== One-stage parallel integration with vigor and rating noise model ========
"""
One-stage parallel integration model with vigor and rating noise.

Parameters:
- θ[1] (d): Drift rate for evidence accumulation across all paths
- θ[2] (θ): Decision threshold (difference between top 2 paths)
- θ[3] (vigor1): Vigor parameter for first stage response time adjustment
- θ[4] (vigor2): Vigor parameter for second stage response time adjustment
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)
- θ[7] (rating_sd): Standard deviation of rating noise added to rewards

Model behavior:
- Rating noise: Adds Gaussian noise to rewards before evidence accumulation
- Single stage: Accumulates evidence for all 4 paths (LL, LR, RL, RR) in parallel
- Decision: Made when difference between top 2 paths exceeds θ
- Vigor effect: Response time adjusted based on difference from maximum reward
- RT1 = T1 + vigor1 * (rmax - r1w)
- RT2 = T2 + vigor2 * (rmax - r2w)
- Note: Incorporates rating uncertainty in addition to vigor effects
"""
function model13(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d              = θ[1]
    θ0             = θ[2]
    vigor1, vigor2 = θ[3], θ[4]
    T1, T2         = θ[5], θ[6]
    rating_sd      = θ[7]
    sigma          = 0.01

    # initialize evidence for four paths
    E = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- sample noisy ratings once per trial ---
    noisy_rewards = rewards .+ rating_sd .* randn(length(rewards))

    # --- first stage ---
    t, rt1, choice1 = 0, 0.0, 0
    max_step = 10000
    idx = 0
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end
        # update evidence for each path
        E[1] += (d * noisy_rewards[1]  + randn()*sigma) + (d * noisy_rewards[3] + randn()*sigma)
        E[2] += (d * noisy_rewards[2]  + randn()*sigma) + (d * noisy_rewards[4] + randn()*sigma)
        E[3] += (d * noisy_rewards[2]  + randn()*sigma) + (d * noisy_rewards[5] + randn()*sigma)

        # check if the leading difference exceeds threshold θ1
        sorted_E = sort(E, rev=true)
        if sorted_E[1] - sorted_E[2] ≥ θ0
            rt1 = t + T1
            # idx==1 → left; idx in {2,3} → right
            idx   = argmax(E)
            choice1 = idx == 1 ? 1 : 2
            break
        end
    end

    # --- second stage ---
    rt2, choice2 = 0.0, 0

    if idx == 1  # LL path
        r1w = rewards[1]
        r2w = rewards[3]
        choice2 = 1
    elseif idx == 2  # RL path
        r1w = rewards[2]
        r2w = rewards[4]
        choice2 = 2
    else  # RR path
        r1w = rewards[2]
        r2w = rewards[5]
        choice2 = 3
    end

    rmax = maximum(rewards)
    rt1 += T1 + vigor1 * (rmax - r1w)
    rt2 += T2 + vigor2 * (rmax - r2w)
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# ======== Two stages with average-based first stage model ========
"""
Two-stage model with average-based first stage decision making.

Parameters:
- θ[1] (d1): Drift rate for first stage evidence accumulation
- θ[2] (d2): Drift rate for second stage evidence accumulation
- θ[3] (θ1): Decision threshold for first stage (difference between L and R)
- θ[4] (θ2): Decision threshold for second stage (difference between remaining options)
- θ[5] (T1): Non-decision time for first stage (baseline response time)
- θ[6] (T2): Non-decision time for second stage (baseline response time)

Model behavior:
- First stage: Accumulates evidence for L vs R using average of leaf values
- L evidence = rewards[1] + 0.5 * (rewards[3] + rewards[4])
- R evidence = rewards[2] + 0.5 * (rewards[5] + rewards[6])
- Decision: Made when difference between L and R exceeds θ1
- Second stage: Only considers paths from chosen side (LL/LR or RL/RR)
- Final choice: Determined by argmax within chosen side
- Note: First stage uses average of leaf values rather than full tree evaluation
"""
function model14(θ::Vector{Float64}, rewards::Vector{Float64})
    # parameters
    d1, d2      = θ[1], θ[2]
    θ1, θ2      = θ[3], θ[4]
    T1, T2      = θ[5], θ[6]
    sigma       = 0.01

    # initialize evidence for four paths
    E1 = zeros(2) # [L, R]
    E2 = zeros(3) # [LL, RL, RR]
    timeout = false

    # --- first stage ---
    t = 0
    rt1 = 0.0
    max_step = 10000
    choice1 = 0
    
    while true
        t += 1
        if t == max_step
            timeout = true
            break
        end

        η_L0 = randn()*sigma
        η_R0 = randn()*sigma
        η_LL = randn()*sigma
        η_RL = randn()*sigma
        η_RR = randn()*sigma

        # accumulate into branch second stage integrators
        E2[1] += d1 * rewards[3] + η_LL
        E2[2] += d1 * rewards[4] + η_RL
        E2[3] += d1 * rewards[5] + η_RR

        # accumulate into first stage integrators
        E1[1] += (d1 * rewards[1]  + η_L0) + d1 * rewards[3] + η_LL
        E1[2] += (d1 * rewards[2]  + η_R0) + 0.5 * ((d1 * rewards[4] + η_RL) + (d1 * rewards[5] + η_RR))

        # check first-stage threshold
        if abs(E1[1] - E1[2]) >= θ1
            rt1    = t + T1
            choice1 = E1[1] > E1[2] ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    # --- second stage ---
    t2 = 0
    rt2, choice2 = 0.0, 0

    if choice1 == 1
        rt2 = T2
        choice2 = 1
    else
        E2 = copy(E2[2:3])
        while true
            t2 += 1
            
            if t2 == max_step
                timeout = true
                break
            end
            
            E2[1] += (d2 * rewards[4] + randn()*sigma)
            E2[2] += (d2 * rewards[5] + randn()*sigma)

            sorted_E2 = sort(E2, rev=true)
            if abs(sorted_E2[1] - sorted_E2[2]) ≥ θ2
                rt2    = t2 + T2
                idx2 = argmax(E2)
                choice2 = idx2 == 1 ? 2 : 3
                break
            end
        end
    end
    
    if timeout
        return (choice1=-1, rt1=-1, choice2=-1, rt2=-1, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end





# - example -
# theta = [8e-5, 6e-5, 0.5, 0.8, 500.0, 500.0]
# theta = [6e-5, 8e-5, 6e-5, 0.3, 0.5, 0.8, 500.0, 500.0]
# theta = [7e-5, 0.6, 1.0, 1.2, 500.0, 500.0, 0.1]
# theta = [7e-5, 8e-5, 6e-5, 0.4, 0.5, 0.8, 500.0, 500.0]

# # Create a Model instance
# model = Model(model10, theta)

# # Create a Trial instance
# trial = Trial(
#     [2.0, -1.0, 2.0, -3.0, 1.0],
#     1,
#     1,
#     0.0,
#     0.0
# )

# # simulate
# result = model.simulate(model.θ, trial.rewards)

# # result containts: choice1, rt1, choice2, rt2, timeout
# println("choice1 = ", result.choice1)  # 1 或 2
# println("rt1     = ", result.rt1)
# println("choice2 = ", result.choice2)  # 1 到 4
# println("rt2     = ", result.rt2)
# println("timeout = ", result.timeout)