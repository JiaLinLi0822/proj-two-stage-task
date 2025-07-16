using JSON

struct Trial
    rewards::Vector{Float64}
    choice1::Int  # Decision at stage 1: 1 for L, 2 for R
    choice2::Int  # Decision at stage 2: 1 for LL, 2 for RL, 3 for RR
    rt1::Float64  # Reaction time for the first decision
    rt2::Float64 # Reaction time for the second decision
end

function load_data_by_subject(filename::String)
    subject_trials = Dict{String, Vector{Trial}}()

    open(filename, "r") do file
        for line in eachline(file)
            data = JSON.parse(line)
            trial = Trial(
                vcat(Float64.(data["value1"]), Float64.(data["value2"])), #[R_L, R_R, R_LL, R_RL, R_RR]
                data["choice1"],
                data["choice2"],
                data["rt1"],
                data["rt2"]
            )
            wid = data["wid"]
            push!(get!(subject_trials, wid, Trial[]), trial)
        end
    end

    return subject_trials  # Dict{String, Vector{Data}}
end

"""
Count trials per participant for BIC calculation.
Returns a dictionary: participant_id => trial_count
"""
function count_trials_per_participant(filename::String)
    trial_counts = Dict{String, Int}()

    open(filename, "r") do file
        for line in eachline(file)
            if !isempty(strip(line))
                data = JSON.parse(line)
                wid = data["wid"]
                trial_counts[wid] = get(trial_counts, wid, 0) + 1
            end
        end
    end

    return trial_counts
end

# - example -
# trials_by_subject = load_data_by_subject("data/Tree1_v3.json")
# println(trials_by_subject)

# # Access subject w11ae0c3's trials
# subject_trials = trials_by_subject["w11ae0c3"]

# # Prepare for parallel fit
# pairs = collect(trials_by_subject)
# results = pmap(x -> fit_subject(x[1], x[2]), pairs)