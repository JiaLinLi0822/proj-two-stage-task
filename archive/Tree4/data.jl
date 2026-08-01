using JSON
using CSV
using DataFrames

"""
Trial struct for Tree3 structure:
- First level: 3 nodes (left, middle, right)
- Second level: 5 leaf nodes
  - Left node → 2 leaves
  - Middle node → 2 leaves  
  - Right node → 1 leaf
- Total: 5 paths, 5 path rewards
"""
struct Trial
    wid::String
    rewards::Vector{Float64}
    choice1::Int  # first-level choice (1,2,3)
    choice2::Int  # second-level choice
    rt1::Float64
    rt2::Float64
    path::Vector{Float64}  # 5 path rewards
end

function load_data(file_path::String)
    trials = Trial[]
    open(file_path, "r") do io
        for line in eachline(io)
            if !isempty(strip(line))
                data = JSON.parse(line)
                push!(trials, Trial(
                    data["wid"],
                    vcat(Float64.(data["value1"]), Float64.(data["value2"])),
                    data["choice1"],
                    data["choice2"],
                    data["rt1"],
                    data["rt2"],
                    data["path_rewards"] # path rewards
                ))
            end
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
                vcat(Float64.(data["value1"]), Float64.(data["value2"])),
                data["choice1"],
                data["choice2"],
                data["rt1"],
                data["rt2"],
                data["path_rewards"] # path rewards
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