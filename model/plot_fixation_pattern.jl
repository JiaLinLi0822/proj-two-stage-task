# plot_fixation_pattern.jl - Plot fixation patterns across multiple trials

using Plots
using Statistics
using Random
using Measures
include("model.jl")

"""
    compute_fixation_duration(trajectory, stage)

Compute fixation duration for each layer in a given stage.
Returns: (first_layer_duration, second_layer_duration)
"""
function compute_fixation_duration(trajectory, stage)
    # First layer nodes: 1, 2 (L, R)
    # Second layer nodes: 3, 4, 5, 6 (LL, LR, RL, RR)
    first_layer_nodes = [1, 2]
    second_layer_nodes = [3, 4, 5, 6]
    
    first_layer_count = 0
    second_layer_count = 0
    
    for step in trajectory
        if step.stage == stage && step.node_observed > 0
            if step.node_observed in first_layer_nodes
                first_layer_count += 1
            elseif step.node_observed in second_layer_nodes
                second_layer_count += 1
            end
        end
    end
    
    return (first_layer_count, second_layer_count)
end

"""
    generate_random_rewards(n_reward_structures; rngseed=nothing)

Generate random reward structures. Each reward is an integer from -4 to 4.
Returns a vector of reward vectors, each of length 6 [L, R, LL, LR, RL, RR].
"""
function generate_random_rewards(n_reward_structures::Int; rngseed::Union{Int,Nothing}=nothing)
    if rngseed !== nothing
        Random.seed!(rngseed)
    end
    
    reward_structures = Vector{Vector{Int}}()
    for i in 1:n_reward_structures
        rewards = rand(-4:4, 6)  # 6 integers from -4 to 4
        push!(reward_structures, rewards)
    end
    
    return reward_structures
end

"""
    run_multiple_trials(rewards, params, n_trials; rngseed=nothing)

Run multiple trials with a single reward structure and collect fixation data.
"""
function run_multiple_trials(rewards::Vector{<:Real}, params::ModelParameters, n_trials::Int; 
                            rngseed::Union{Int,Nothing}=nothing)
    stage1_first = Float64[]
    stage1_second = Float64[]
    stage2_first = Float64[]
    stage2_second = Float64[]
    
    for i in 1:n_trials
        seed = rngseed === nothing ? nothing : rngseed + i - 1
        result = run_model(Float64.(rewards), params; rngseed=seed)
        
        # Compute fixation durations for each stage
        (f1, s1) = compute_fixation_duration(result.trajectory, 1)
        (f2, s2) = compute_fixation_duration(result.trajectory, 2)
        
        push!(stage1_first, f1)
        push!(stage1_second, s1)
        push!(stage2_first, f2)
        push!(stage2_second, s2)
    end
    
    return (stage1_first, stage1_second, stage2_first, stage2_second)
end

"""
    run_multiple_reward_structures(reward_structures, params, n_trials_per_structure; rngseed=nothing, show_progress=true)

Run trials across multiple reward structures and aggregate fixation data.
"""
function run_multiple_reward_structures(reward_structures::Vector{Vector{Int}}, 
                                       params::ModelParameters, 
                                       n_trials_per_structure::Int;
                                       rngseed::Union{Int,Nothing}=nothing,
                                       show_progress::Bool=true)
    all_stage1_first = Float64[]
    all_stage1_second = Float64[]
    all_stage2_first = Float64[]
    all_stage2_second = Float64[]
    
    n_structures = length(reward_structures)
    start_time = time()
    
    for (idx, rewards) in enumerate(reward_structures)
        base_seed = rngseed === nothing ? nothing : rngseed + (idx - 1) * n_trials_per_structure
        (s1_f, s1_s, s2_f, s2_s) = run_multiple_trials(rewards, params, n_trials_per_structure; 
                                                       rngseed=base_seed)
        append!(all_stage1_first, s1_f)
        append!(all_stage1_second, s1_s)
        append!(all_stage2_first, s2_f)
        append!(all_stage2_second, s2_s)
        
        if show_progress
            elapsed = time() - start_time
            progress = idx / n_structures
            avg_time_per_structure = elapsed / idx
            remaining_structures = n_structures - idx
            estimated_remaining = avg_time_per_structure * remaining_structures
            
            println("Progress: $idx/$n_structures structures ($(round(progress*100, digits=1))%) | ",
                   "Elapsed: $(round(elapsed, digits=1))s | ",
                   "Est. remaining: $(round(estimated_remaining, digits=1))s")
        end
    end
    
    total_time = time() - start_time
    if show_progress
        println("\nCompleted in $(round(total_time, digits=1)) seconds")
        println("Average time per structure: $(round(total_time/n_structures, digits=2))s")
        println("Average time per trial: $(round(total_time/(n_structures*n_trials_per_structure), digits=3))s")
    end
    
    return (all_stage1_first, all_stage1_second, all_stage2_first, all_stage2_second)
end

"""
    plot_fixation_pattern(stage1_first, stage1_second, stage2_first, stage2_second;
                         title="Fixation Pattern", output_file=nothing)

Plot fixation duration by stage and layer.
"""
function plot_fixation_pattern(stage1_first, stage1_second, stage2_first, stage2_second;
                              title="Fixation Pattern", output_file=nothing)
    
    # Compute means and standard errors
    mean_s1_f = mean(stage1_first)
    sem_s1_f = std(stage1_first) / sqrt(length(stage1_first))
    mean_s1_s = mean(stage1_second)
    sem_s1_s = std(stage1_second) / sqrt(length(stage1_second))
    
    mean_s2_f = mean(stage2_first)
    sem_s2_f = std(stage2_first) / sqrt(length(stage2_first))
    mean_s2_s = mean(stage2_second)
    sem_s2_s = std(stage2_second) / sqrt(length(stage2_second))
    
    # Prepare data for grouped bar chart
    stages = ["Stage 1", "Stage 2"]
    first_layer_means = [mean_s1_f, mean_s2_f]
    second_layer_means = [mean_s1_s, mean_s2_s]
    first_layer_sems = [sem_s1_f, sem_s2_f]
    second_layer_sems = [sem_s1_s, sem_s2_s]
    
    # Create grouped bar plot
    x_pos = [1, 2]  # Stage positions
    bar_width = 0.35
    
    plt = plot(size=(600, 400), dpi=300)
    
    # First layer bars with error bars
    x1 = x_pos .- bar_width/2
    bar!(plt, x1, first_layer_means, 
        width=bar_width, 
        label="First Layer",
        color=:steelblue,
        alpha=0.7)
    
    # Add error bars for first layer
    for i in 1:length(x1)
        x = x1[i]
        y = first_layer_means[i]
        err = first_layer_sems[i]
        plot!(plt, [x, x], [y - err, y + err], 
              color=:steelblue, linewidth=2, label="")
        plot!(plt, [x - bar_width/8, x + bar_width/8], [y - err, y - err],
              color=:steelblue, linewidth=2, label="")
        plot!(plt, [x - bar_width/8, x + bar_width/8], [y + err, y + err],
              color=:steelblue, linewidth=2, label="")
    end
    
    # Second layer bars with error bars
    x2 = x_pos .+ bar_width/2
    bar!(plt, x2, second_layer_means,
        width=bar_width,
        label="Second Layer",
        color=:coral,
        alpha=0.7)
    
    # Add error bars for second layer
    for i in 1:length(x2)
        x = x2[i]
        y = second_layer_means[i]
        err = second_layer_sems[i]
        plot!(plt, [x, x], [y - err, y + err], 
              color=:coral, linewidth=2, label="")
        plot!(plt, [x - bar_width/8, x + bar_width/8], [y - err, y - err],
              color=:coral, linewidth=2, label="")
        plot!(plt, [x - bar_width/8, x + bar_width/8], [y + err, y + err],
              color=:coral, linewidth=2, label="")
    end
    
    # Customize plot
    plot!(plt,
        xlabel="Stage",
        ylabel="Fixation Duration (observations)",
        title=title,
        xticks=(x_pos, stages),
        legend=:topright,
        grid=true,
        ylims=(0, maximum([first_layer_means; second_layer_means]) * 1.2),
        bottom_margin=5mm,
        left_margin=5mm)
    
    if output_file !== nothing
        savefig(plt, output_file)
        println("Plot saved to: ", output_file)
    end
    
    return plt
end

# Main execution - run when script is executed directly
# This will run in IDE or when the file is included
if true  # Always run when included
    # Default parameters
    n_reward_structures = 20  # Number of different reward structures
    n_trials_per_structure = 10  # Number of trials per reward structure
    total_trials = n_reward_structures * n_trials_per_structure
    
    model_params = ModelParameters(;
        mu0=0.0,
        tau0=1.0,
        sigma=0.6,
        gamma=1.0,
        H_thresh1=0.85,
        H_thresh2=0.45,
        Tsoft=1.0,
        eps_nodesel=0.05,
        dt=0.01,
        alpha=1.0,
        p_stay=0.75,
        lambda=1.2,
        use_genz=true
    )
    
    println("Generating ", n_reward_structures, " random reward structures...")
    reward_structures = generate_random_rewards(n_reward_structures; rngseed=42)
    
    println("Reward structures (first 5):")
    for (i, rewards) in enumerate(reward_structures[1:min(5, length(reward_structures))])
        println("  Structure $i: ", rewards, " [L, R, LL, LR, RL, RR]")
    end
    
    println("\nRunning ", total_trials, " trials across ", n_reward_structures, " reward structures...")
    println("  (", n_trials_per_structure, " trials per structure)")
    
    (s1_f, s1_s, s2_f, s2_s) = run_multiple_reward_structures(reward_structures, model_params, 
                                                               n_trials_per_structure; 
                                                               rngseed=123)
    
    println("\nSummary statistics (across all reward structures):")
    println("Stage 1 - First Layer:  mean = ", round(mean(s1_f), digits=2), 
            ", SEM = ", round(std(s1_f)/sqrt(length(s1_f)), digits=2))
    println("Stage 1 - Second Layer: mean = ", round(mean(s1_s), digits=2),
            ", SEM = ", round(std(s1_s)/sqrt(length(s1_s)), digits=2))
    println("Stage 2 - First Layer:  mean = ", round(mean(s2_f), digits=2),
            ", SEM = ", round(std(s2_f)/sqrt(length(s2_f)), digits=2))
    println("Stage 2 - Second Layer: mean = ", round(mean(s2_s), digits=2),
            ", SEM = ", round(std(s2_s)/sqrt(length(s2_s)), digits=2))
    
    # Plot
    plt = plot_fixation_pattern(s1_f, s1_s, s2_f, s2_s;
                               title="Fixation Pattern (n=$total_trials trials, $n_reward_structures reward structures)",
                               output_file="fixation_pattern.png")
    
    display(plt)
end

