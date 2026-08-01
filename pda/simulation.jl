include("data.jl")
include("model6.jl")

function simulate_trials(trials::Vector{Trial}, params::Vector{Float64}; n_sims::Int=1000)
    
    results = Trial[]
    for trial in trials
        for _ in 1:n_sims
            sim_result = model6(params, trial.rewards)
            
            sim_trial = Trial(
                trial.wid,
                trial.rewards,
                sim_result.choice1,
                sim_result.choice2,
                sim_result.rt1,
                sim_result.rt2,
                trial.path
            )
            push!(results, sim_trial)
        end
    end
    
    return results
end
