using CSV
using DataFrames
using JSON
using Random
using Statistics

include("model.jl")
include("model_configs.jl")
include("data.jl")
include("add_info.jl")


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
                        param_file::String = "julia/results_0714.csv",
                        trial_file::String = "data/Tree2_v3.json",
                        output_file::String = "data/simulated_$(model_name).json",
                        n_simulations::Int = 1,
                        random_seed::Int = 42)
    
    Random.seed!(random_seed)
    
    config = get_model_config(model_name)
    model_function = config.model_function
    
    println("="^60)
    println("Simulating trials using $model_name")
    println("Description: $(config.description)")
    println("="^60)
    
    println("Loading fitted parameters from $param_file...")
    param_dict = load_fitted_parameters(param_file, model_name)
    
    println("Loading trial data from $trial_file...")
    trials_by_wid = load_data_by_subject(trial_file)
    
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
        
        for trial in subject_trials
            rewards = trial.rewards
            
            for sim_idx in 1:n_simulations
                result = model_function(theta, rewards)
                
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
                
                # Calculate difficulties
                diff1 = round(calculate_diff1(trial.path))
                diff2 = choice1_sim == -1 ? -1.0 : calculate_diff2([trial.rewards[3], trial.rewards[4], trial.rewards[5], trial.rewards[6]], choice1_sim)
                difficulty = diff1  # Overall trial difficulty
                
                result_entry = Dict(
                    "model" => model_name,
                    "simulation_id" => sim_idx,
                    "wid" => wid,
                    "rewards" => trial.path,
                    "value1" => [trial.rewards[1], trial.rewards[2]],
                    "value2" => [trial.rewards[3], trial.rewards[4], trial.rewards[5], trial.rewards[6]],
                    "path" => trial.path,
                    "choice1" => choice1_sim,
                    "choice2" => choice2_sim,
                    "rt1" => rt1_sim,
                    "rt2" => rt2_sim,
                    "rt" => rt_total,
                    "timeout" => result.timeout,
                    "diff1" => diff1,
                    "diff2" => diff2,
                    "difficulty" => difficulty
                )
                
                push!(all_results, result_entry)
                total_trials += 1
            end
        end
        
        processed_subjects += 1
        if processed_subjects % 10 == 0
            println("Processed $processed_subjects subjects...")
        end
    end
    
    println("Saving simulated trials to $output_file...")
    output_dir = dirname(output_file)
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    open(output_file, "w") do file
        for result in all_results
            println(file, JSON.json(result))
        end
    end
    
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



# Simulation Configuration
MODEL_NAME = "model2"
PARAM_FILE = "Tree2/results/pda/model2_pda_BADS_20251003_171731.csv"
TRIAL_FILE = "Tree2/data/Tree2_v3.json"
OUTPUT_FILE = "Tree2/data/pda/$(MODEL_NAME)_pda.json"

# Run simulation
println("Starting simulation with model: $MODEL_NAME")
results = simulate_trials(MODEL_NAME; 
                          param_file=PARAM_FILE,
                          trial_file=TRIAL_FILE,
                          output_file=OUTPUT_FILE) 


