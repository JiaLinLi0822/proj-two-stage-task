using JSON
using CSV
using DataFrames

struct Trial
    wid::String   # Participant ID
    rewards::Vector{Float64}
    choice1::Int  # Decision at stage 1: 1 for L, 2 for R
    choice2::Int  # Decision at stage 2: 1 for LL, 2 for LR, 3 for RL
    rt1::Float64  # Reaction time for the first decision
    rt2::Float64  # Reaction time for the second decision
    path::Vector{Float64}  # Path of the trial
end

function load_data(filename::String)
    trials = Vector{Trial}()
    open(filename, "r") do file
        for line in eachline(file)
            data = JSON.parse(line)
            push!(trials, Trial(
                data["wid"],
                vcat(Float64.(data["value1"]), Float64.(data["value2"])), #[R_L, R_R, R_LL, R_LR, R_RL]
                data["choice1"],
                data["choice2"],
                data["rt1"],
                data["rt2"],
                data["rewards"] # path values
            ))
        end
    end
    return trials
end

function load_data_by_subject(filename::String)
    subject_trials = Dict{String, Vector{Trial}}()

    open(filename, "r") do file
        for line in eachline(file)
            data = JSON.parse(line)
            trial = Trial(
                data["wid"],
                vcat(Float64.(data["value1"]), Float64.(data["value2"])), #[R_L, R_R, R_LL, R_LR, R_RL]
                data["choice1"],
                data["choice2"],
                data["rt1"],
                data["rt2"],
                data["rewards"] # path values
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


"""
Load fitted parameters from CSV file and organize by subject and model.
Returns a dictionary: subject_id => parameter_vector
"""
function load_fitted_parameters(param_file::String, model_name::String)
    if !isfile(param_file)
        error("Parameter file not found: $param_file")
    end
    
    param_df = CSV.read(param_file, DataFrame)
    config = get_model_config(model_name)
    param_names = config.param_names
    
    println("Model $model_name requires parameters: $(param_names)")
    println("CSV file has columns: $(names(param_df))")
    
    param_dict = Dict{String, Vector{Float64}}()
    
    for row in eachrow(param_df)
        wid = string(row.wid)
        params = Float64[]
        
        for param_name in param_names
            col_symbol = Symbol(param_name)
            if haskey(row, col_symbol)
                raw = row[col_symbol]
                val = raw isa AbstractString ? parse(Float64, raw) : Float64(raw)
                push!(params, val)
            else
                error("Parameter $param_name not found in CSV for model $model_name. Available columns: $(names(param_df))")
            end
        end
        
        param_dict[wid] = params
    end
    
    println("Loaded parameters for $(length(param_dict)) subjects")
    return param_dict
end


# - example -
# trials_by_subject = load_data_by_subject("julia/w11ae0c3.json")
# println(trials_by_subject)

# # Access subject w11ae0c3's trials
# subject_trials = trials_by_subject["w11ae0c3"]

# # Prepare for parallel fit
# pairs = collect(trials_by_subject)
# results = pmap(x -> fit_subject(x[1], x[2]), pairs)