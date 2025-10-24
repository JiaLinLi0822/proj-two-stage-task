using CSV
using DataFrames
using JSON
using Random
using Statistics
include("model.jl")
include("model_configs.jl")
include("data.jl")

"""
Comprehensive trial simulation system supporting all 14 models.
This script uses the model configuration system to automatically handle
parameter structures and bounds for any available model.
"""

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
    param_names = config.param_names  # Use the dedicated param_names from model_configs
    
    println("Model $model_name requires parameters: $(param_names)")
    println("CSV file has columns: $(names(param_df))")
    
    # Map parameter names to CSV column names using model_configs param_names
    param_dict = Dict{String, Vector{Float64}}()
    
    for row in eachrow(param_df)
        wid = string(row.wid)
        params = Float64[]
        
        # Use the exact param_names from model_configs
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

"""
Simulate trials for a specific model using fitted parameters.

Arguments:
- model_name: Name of the model to simulate (e.g., "model1", "model2", etc.)
- param_file: CSV file containing fitted parameters
- trial_file: JSON file containing trial data
- output_file: Output file for simulated results
- n_simulations: Number of simulation runs per trial (default: 1)
"""
function simulate_trials(model_name::String; 
                        param_file::String,
                        trial_file::String,
                        output_file::String,
                        n_simulations::Int = 1,
                        random_seed::Int = 42)
    
    # Set random seed for reproducibility
    Random.seed!(random_seed)
    
    # Validate model
    if !haskey(MODEL_CONFIGS, model_name)
        available_models = join(keys(MODEL_CONFIGS), ", ")
        error("Model '$model_name' not found. Available models: $available_models")
    end
    
    config = get_model_config(model_name)
    model_function = config.model_function
    
    println("="^60)
    println("Simulating trials using $model_name")
    println("Description: $(config.description)")
    println("="^60)
    
    # Load fitted parameters
    println("Loading fitted parameters from $param_file...")
    param_dict = load_fitted_parameters(param_file, model_name)
    
    # Load trial data
    println("Loading trial data from $trial_file...")
    trials = []
    open(trial_file, "r") do file
        for line in eachline(file)
            if !isempty(strip(line))
                push!(trials, JSON.parse(line))
            end
        end
    end
    
    # Group trials by subject
    println("Grouping trials by subject...")
    trials_by_wid = Dict{String, Vector{Any}}()
    for trial in trials
        wid = trial["wid"]
        if !haskey(trials_by_wid, wid)
            trials_by_wid[wid] = []
        end
        push!(trials_by_wid[wid], trial)
    end
    
    # Simulate trials
    all_results = []
    processed_subjects = 0
    total_trials = 0
    timeout_count = 0
    
    println("Simulating trials...")
    
    for (wid, subject_trials) in trials_by_wid
        if !haskey(param_dict, wid)
            continue  # skip if no fitted parameters for this subject
        end
        
        theta = param_dict[wid]
        
        for entry in subject_trials
            # Prepare reward structure: [R_L, R_R, R_LL, R_RL, R_RR]
            rewards = Float64[
                entry["value1"][1],  # R_L
                entry["value1"][2],  # R_R  
                entry["value2"][1],  # R_LL
                entry["value2"][2],  # R_RL
                entry["value2"][3],  # R_RR
            ]
            
            # Run multiple simulations if requested
            for sim_idx in 1:n_simulations
                # Simulate trial using the specified model
                result = model_function(theta, rewards)
                
                # Handle timeout cases
                if result.timeout
                    choice1_sim = -1
                    choice2_sim = -1
                    rt1_sim = -1
                    rt2_sim = -1
                    rt_total = -1
                    timeout_count += 1
                else
                    choice1_sim = result.choice1
                    choice2_sim = result.choice2
                    rt1_sim = round(Int, result.rt1)
                    rt2_sim = round(Int, result.rt2)
                    rt_total = rt1_sim + rt2_sim
                end
                
                # Store results
                result_entry = Dict(
                    "wid" => wid,
                    "rewards" => entry["rewards"],
                    "value1" => entry["value1"],
                    "value2" => entry["value2"],
                    "choice1" => choice1_sim,
                    "choice2" => choice2_sim,
                    "rt1" => rt1_sim,
                    "rt2" => rt2_sim,
                    "rt" => rt_total,
                    "timeout" => result.timeout
                )
                
                push!(all_results, result_entry)
                total_trials += 1
            end
        end
        
        processed_subjects += 1
        if processed_subjects % 5 == 0
            println("Processed $processed_subjects subjects...")
        end
    end
    
    # Save results
    println("Saving simulated trials to $output_file...")
    open(output_file, "w") do file
        JSON.print(file, all_results, 2)
    end
    
    # Print summary
    println("="^60)
    println("Simulation Summary")
    println("="^60)
    println("Model: $model_name")
    println("Subjects processed: $processed_subjects")
    println("Total trials simulated: $total_trials")
    println("Timeouts: $timeout_count ($(round(100*timeout_count/total_trials, digits=2))%)")
    println("Output saved to: $output_file")
    println("="^60)
    
    return all_results
end


"""
List all available models with their descriptions.
"""
function show_available_models()
    list_models()
end

# Command line interface
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    args = ARGS
    
    if length(args) == 0
        println("Usage:")
        println("  julia simulate_trials.jl <model_name> [param_file] [trial_file] [output_file]")
        println("  julia simulate_trials.jl list  # Show available models")
        println("")
        println("Examples:")
        println("  julia simulate_trials.jl model1")
        println("  julia simulate_trials.jl model2 julia/custom_results.csv")
        println("  julia simulate_trials.jl model11 julia/results_0708.csv data/Tree2_v3.json output/sim_model11.json")
        println("")
        show_available_models()
        exit(1)
    end
    
    if args[1] == "list"
        show_available_models()
        exit(0)
    end
    
    # Extract arguments
    model_name = args[1]
    param_file = length(args) >= 2 ? args[2] : "julia/results_0714.csv"
    trial_file = length(args) >= 3 ? args[3] : "data/Tree2_v3.json"
    output_file = length(args) >= 4 ? args[4] : "data/simulated_$(model_name).json"
    
    # Run simulation
    println("Starting simulation with model: $model_name")
    results = simulate_trials(model_name; 
                            param_file=param_file,
                            trial_file=trial_file,
                            output_file=output_file)

end 

### Use example:

# julia Tree1/simulation.jl model1 Tree1/results/model1_20250716_221644.csv data/Tree1_v3.json data/Tree1_sim/simulate_model1.json
# julia Tree1/simulation.jl model3 Tree1/results/model3_20250717_005240.csv data/Tree1_v3.json data/Tree1_sim/simulate_model3.json
# julia Tree1/simulation.jl model6 Tree1/results/model6_20250717_032409.csv data/Tree1_v3.json data/Tree1_sim/simulate_model6.json
# julia Tree1/simulation.jl model7 Tree1/results/model7_20250717_033845.csv data/Tree1_v3.json data/Tree1_sim/simulate_model7.json


