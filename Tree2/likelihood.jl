include("ibs.jl")
include("data.jl")
include("model.jl")
include("pda.jl")

# ======== IBS Likelihood Functions ========
function max_rt1(t::Trial)
    # Maximum reasonable reaction time (in ms)
    return 10000.0
end

function max_rt2(t::Trial)
    # Maximum reasonable reaction time (in ms)
    return 10000.0
end

function is_hit((choice1, rt1, choice2, rt2), t::Trial, rt_tol1, rt_tol2)
    t.choice1 == choice1 && abs(rt1 - t.rt1) ≤ rt_tol1 &&
    t.choice2 == choice2 && abs(rt2 - t.rt2) ≤ rt_tol2
end

function sample_choice_rt(m::Model, t::Trial, ε)
    if rand() < ε
        # Lapse trial - random response
        choice1 = rand(1:2)
        if choice1 == 1
            choice2 = rand(1:2)
        else
            choice2 = rand(3:4)
        end
        rt1 = rand(100.0:max_rt1(t))  # Random RT within reasonable range
        rt2 = rand(100.0:max_rt2(t))
        return (choice1, rt1, choice2, rt2)
    else
        sim = simulate(m, t)
        sim.timeout && return (-1, -1, -1, -1)
        return (sim.choice1, sim.rt1, sim.choice2, sim.rt2)
    end
end

function fixed_loglike(m, t::Trial; ε=.05, rt_tol1=0, rt_tol2=0, N=10000)
    hits = 0
    for i in 1:N
        if is_hit(sample_choice_rt(m, t, ε), t, rt_tol1, rt_tol2)
            hits +=1
        end
    end
    log((hits + 1) / (N + 1))
end

function chance_loglike(trials; rt_tol1=0, rt_tol2=0)
    mapreduce(+, trials) do t
        n_within_tol1 = 1 + min(max_rt1(t), t.rt1 + rt_tol1) - max(1, t.rt1 - rt_tol1)
        n_within_tol2 = 1 + min(max_rt2(t), t.rt2 + rt_tol2) - max(1, t.rt2 - rt_tol2)
        # log(0.5) + log(0.5) + log(n_within_tol1 / max_rt1(t)) + log(n_within_tol2 / max_rt2(t))
        
        # Clamp to ensure positive probabilities
        prob1 = clamp(n_within_tol1 / max_rt1(t), 1e-10, 1.0)
        prob2 = clamp(n_within_tol2 / max_rt2(t), 1e-10, 1.0)
        
        log(0.5) + log(0.5) + log(prob1) + log(prob2)
    end
end

function ibs_loglike(m::Model, trials::Vector{Trial}; repeats=1, max_iter=1000, ε=.05, rt_tol1=0, rt_tol2=0, min_multiplier=0.8)
    neg_logp_threshold = min_multiplier * (-chance_loglike(trials; rt_tol1, rt_tol2))
    result = ibs(trials; repeats, max_iter, neg_logp_threshold) do t
        is_hit(sample_choice_rt(m, t, ε), t, rt_tol1, rt_tol2)
    end
    return result
end




# ======== PDA Likelihood Functions ========

"""
    pda_loglike_single_trial(θ, trial, model_func; J=2000, kwargs...)

Calculate PDA-based log-likelihood for a single trial using the specified model function.

Args:
- θ: Parameter vector
- trial: Trial struct
- model_func: Model function to use
- J: Minimum number of simulations (mapped to min_sims)
- min_sims: Minimum number of simulations to perform (default: 1000)
- min_matching: Minimum number of matching samples required (default: 100)
- max_sims: Maximum number of simulations to prevent infinite loops (default: 1000000)
- kwargs: Additional arguments for PDA functions

Returns:
- log-likelihood value for the trial
"""
function pda_loglike_single_trial(θ::Vector{Float64}, trial::Trial, model_func::Function;
                                   J::Int=1000,
                                   min_sims::Int=1000,
                                   min_matching::Int=100,
                                   max_sims::Int=10000,
                                   kde_mode::Symbol=:gaussian,       # :product or :gaussian
                                   bw_rule::Symbol=:silverman,       # for :gaussian
                                   logRT::Bool=true,
                                   eps_floor::Float64=1e-16,
                                   lambda::Float64=1.0)

    results = pda_sampler(trial, θ, model_func; 
                         min_sims=max(min_sims, J),
                         min_matching=min_matching,
                         max_sims=max_sims)
    pairs = [(1,1), (1,2), (2,3), (2,4)]  # Tree2
    spdf = build_mixed2d_spdf(results, trial;
                              pairs=pairs, bw_rule=bw_rule, logRT=logRT,
                              eps_floor=eps_floor)
    ll, used_eps = mixed2d_logpdf(spdf, trial, lambda)
    return ll, used_eps
end

"""
    pda_loglike(θ, trials, model_func; J=2000, kwargs...)

Calculate PDA-based log-likelihood for a vector of trials.

Args:
- θ: Parameter vector
- trials: Vector of Trial structs
- model_func: Model function to use
- J: Minimum number of simulations (mapped to min_sims)
- min_sims: Minimum number of simulations to perform (default: 1000)
- min_matching: Minimum number of matching samples required (default: 100)
- max_sims: Maximum number of simulations to prevent infinite loops (default: 1000000)
- kwargs: Additional arguments for PDA functions

Returns:
- Total log-likelihood across all trials
"""
function pda_loglike(θ::Vector{Float64}, trials::Vector{Trial}, model_func::Function;
                    J::Int=1000, 
                    min_sims::Int=1000,
                    min_matching::Int=100,
                    max_sims::Int=10000,
                    kde_mode::Symbol=:gaussian, bw_rule::Symbol=:silverman,
                    logRT::Bool=true, eps_floor::Float64=1e-16, lambda::Float64=1.0)
    total_ll = 0.0
    eps_floor_count = 0
    for trial in trials
        ll, used_eps = pda_loglike_single_trial(θ, trial, model_func; 
                                            J=J,
                                            min_sims=min_sims,
                                            min_matching=min_matching,
                                            max_sims=max_sims,
                                            kde_mode=kde_mode, 
                                            bw_rule=bw_rule,
                                            logRT=logRT,
                                            eps_floor=eps_floor,
                                            lambda=lambda)
        total_ll += ll
        if used_eps
            eps_floor_count += 1
        end
    end
    return total_ll, eps_floor_count, length(trials)
end

