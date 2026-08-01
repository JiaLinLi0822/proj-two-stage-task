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
function model6(φ::Vector{Float64}, rewards::Vector{Float64})
    
    d1, d2      = φ[1], φ[2]
    θ1, θ2      = φ[3], φ[4]
    T1, T2      = φ[5], φ[6]
    sigma       = 0.01

    E1 = zeros(2)
    timeout = false

    # --- first stage ---
    t = 0
    rt1 = 0.0
    choice1 = 0
    
    while true
        t += 1

        E1[1] += (d1 * rewards[1]  + randn()*sigma)
        E1[2] += (d1 * rewards[2]  + randn()*sigma)

        if abs(E1[1] - E1[2]) >= θ1
            rt1    = t + T1
            choice1 = E1[1] > E1[2] ? 1 : 2
            break
        end
    end

    if timeout
        return (choice1=0, rt1=0, choice2=0, rt2=0, timeout=true)
    end

    # --- second stage ---
    t2 = 0
    rt2 = 0.0
    E2 = zeros(2) # [LL, LR] or [RL, RR]
    choice2 = 0

    while true
        t2 += 1

        if choice1 == 1 
            E2[1] += (d2 * rewards[3] + randn()*sigma)
            E2[2] += (d2 * rewards[4] + randn()*sigma)
        else               
            E2[1] += (d2 * rewards[5] + randn()*sigma)
            E2[2] += (d2 * rewards[6] + randn()*sigma)
        end

        if abs(E2[1] - E2[2]) ≥ θ2
            rt2    = t2 + T2
            choice2_idx = E2[1] > E2[2] ? 1 : 2

            if choice1 == 1
                choice2 = choice2_idx 
            else
                choice2 = choice2_idx + 2
            end
            break
        end
    end
    
    if timeout
        return (choice1=0, rt1=0, choice2=0, rt2=0, timeout=true)
    end

    return (choice1=choice1, rt1=rt1, choice2=choice2, rt2=rt2, timeout=false)
end