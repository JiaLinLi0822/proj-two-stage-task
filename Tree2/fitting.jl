# fitting.jl

using Distributed
using Dates


if myid() == 1 && nprocs() == 1 && !haskey(ENV, "SLURM_JOB_ID")
    n = try
        parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")) - 1
    catch
        0
    end
    n = max(n, 1)
    @info "Adding $n local workers (non-SLURM environment)"
    addprocs(n)
end

@everywhere begin
    using LinearAlgebra
    LinearAlgebra.BLAS.set_num_threads(1)

    include("ibs.jl")
    include("model.jl")
    include("likelihood.jl")
    include("data.jl")
    include("bads.jl")
    include("model_configs.jl")
    include("pda.jl")
    include("fpt.jl")

    using JSON, DataFrames, CSV, Logging, Random, BlackBoxOptim
    disable_logging(Logging.Warn)
end

@everywhere function fit_subject(wid, trials, model_name::String, likelihood_method::String="ibs";
                                kde_mode::Symbol=:gaussian, bw_rule::Symbol=:silverman, J::Int=500, 
                                min_sims::Int=1000, min_matching::Int=100, max_sims::Int=10000,
                                eps_floor::Float64=1e-16, lambda::Float64=1.0, optimizer::Symbol=:DE)
    
    # Get model configuration
    config = get_model_config(model_name)
    
    # Get hard bounds
    box = config.hard_bounds
    param_names = collect(keys(box.dims))
    n = n_free(box)

    lbs = zeros(n)
    ubs = ones(n)

    # Get plausible bounds
    pbox = config.plausible_bounds
    lower_dict = Dict(name => pbox.dims[name][1] for name in param_names)
    upper_dict = Dict(name => pbox.dims[name][2] for name in param_names)

    plbs = apply(box, lower_dict)
    pubs = apply(box, upper_dict)

    # Get initial parameters
    x0 = apply(box, config.initial_params)
    
    optimizer_label = optimizer == :DE ? "DE" : "BADS"
    
    println("Worker $(myid()): Starting $optimizer_label optimization for subject $wid using $model_name")
    
    # Tracker
    eval_count = Ref(0)
    
    # Objective function
    function objective_function(x_unit)
        try
            eval_count[] += 1
            if eval_count[] % 100 == 0
                println("Worker $(myid()): Subject $wid - Evaluation $(eval_count[])")
            end
        
            x = Float64.(x_unit)
            θ_dict = box(x)
            θ = [θ_dict[name] for name in param_names]
            
            if likelihood_method == "ibs"
                model = Model(config.model_function, θ)
        
                res = ibs_loglike(model, trials;
                                  repeats  = 10,
                                  max_iter = 1000,
                                  ε        = 0.05,
                                  rt_tol1  = 1000,
                                  rt_tol2  = 1000,
                                  min_multiplier = 0.8)
            
                neg_ll = res.neg_logp
            elseif likelihood_method == "pda"

                ll, eps_floor_count, n_trials =  pda_loglike(θ, trials, 
                                  config.model_function;
                                  J=J,
                                  min_sims=min_sims,
                                  min_matching=min_matching,
                                  max_sims=max_sims,
                                  kde_mode=kde_mode, 
                                  bw_rule=bw_rule, 
                                  eps_floor=eps_floor,
                                  lambda=lambda)
                neg_ll = -ll
                
                # Report eps_floor usage periodically
                if eval_count[] % 100 == 0 || eval_count[] == 1
                    eps_floor_pct = round(100.0 * eps_floor_count / n_trials, digits=1)
                    println("Worker $(myid()): Subject $wid - Eval $(eval_count[]) | eps_floor used: $eps_floor_count/$n_trials ($eps_floor_pct%)")
                end
            elseif likelihood_method == "analytical"
                analytical_eps_floor = 1e-16
                total_ll = 0.0
                for trial in trials
                    ll = loglik_trial_stagewise(trial, θ; eps_floor=analytical_eps_floor)
                    total_ll += ll
                end
                neg_ll = -total_ll
                if !isfinite(neg_ll)
                    return 1e6
                end
            else
                error("Unknown likelihood method: $likelihood_method. Use 'ibs', 'pda', or 'analytical'.")
            end
    
            return Float64(neg_ll)
    
        catch e
            @error "Worker $(myid()): Exception for $wid: $(e)"
            return 1e6
        end
    end
    
    try
        optimizer_label = optimizer == :DE ? "DE" : "BADS"
        if optimizer == :BADS
            bads_result = optimize_bads(objective_function;
                x0 = x0,
                lower_bounds = lbs,
                upper_bounds = ubs, 
                plausible_lower_bounds = plbs,
                plausible_upper_bounds = pubs,
                max_fun_evals = 1000,
                uncertainty_handling = false,
                specify_target_noise = false,
            )
            result_dict = get_result(bads_result)
            xopt_unit_any = result_dict["x"]
            fopt = result_dict["fval"]
            xopt_unit = Float64.(xopt_unit_any)

        elseif optimizer == :DE
            search_range = [(plbs[i], pubs[i]) for i in 1:n]
            result = bboptimize(objective_function;
                Method = :de_rand_1_bin,
                LowerBounds = lbs,
                UpperBounds = ubs,
                SearchRange = search_range,
                NumDimensions = n,
                MaxFuncEvals = 2000,
                TraceMode = :compact,
                # TraceInterval = 50,
                FitnessTolerance = 1e-2,
            )
            xopt_unit = Float64.(best_candidate(result))
            fopt = Float64(best_fitness(result))
        end

        real_xopt     = box(xopt_unit)
        xopt = [real_xopt[name] for name in param_names]

        println("Worker $(myid()): $optimizer completed for subject $wid ($(eval_count[]) evaluations)")
        println("Worker $(myid()): Subject $wid - $optimizer_label - Final θ = $xopt, negLL = $fopt")
        
        return wid, xopt, fopt, param_names, optimizer_label
        
    catch e
        @error "Worker $(myid()): $optimizer_label failed for subject $wid" exception=(e, catch_backtrace())
        return wid, x0, 1e6, param_names, optimizer_label
    end
end

"""
Run model fitting for all subjects using the specified model.
"""
function run_model_fitting(model_name::String; 
                          data_file::String = "Tree2/data/Tree2_v3.json",
                          output_file::Union{String, Nothing} = nothing,
                          likelihood_method::String = "ibs",
                          kde_mode::Symbol = :gaussian,
                          bw_rule::Symbol = :silverman,
                          J::Int = 1000,
                          min_sims::Int = 1000,
                          min_matching::Int = 100,
                          max_sims::Int = 10000,
                          eps_floor::Float64 = 1e-16,
                          lambda::Float64 = 1.0,
                          optimizer::Symbol = :DE)
    
    config = get_model_config(model_name)
    
    if output_file === nothing
        if likelihood_method == "pda"
            output_file = "Tree2/results/pda/$(model_name)_$(likelihood_method)_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
        elseif likelihood_method == "analytical"
            output_file = "Tree2/results/analytical/$(model_name)_$(likelihood_method)_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
        else
            output_file = "Tree2/results/ibs/$(model_name)_$(likelihood_method)_$(optimizer)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
        end
    end

    output_dir = dirname(output_file)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    println("="^60)
    println("Model Fitting Configuration")
    println("="^60)
    println("Model: $model_name")
    println("Description: $(config.description)")
    println("Likelihood method: $likelihood_method")
    println("Optimizer: $optimizer")
    if likelihood_method == "pda"
        println("PDA Configuration:")
        println("  KDE mode: $kde_mode")
        println("  Bandwidth rule: $bw_rule")
        println("  Eps floor: $eps_floor")
        println("  Simulations per trial: $J")
    elseif likelihood_method == "analytical"
        println("Analytical Configuration:")
        println("  Using first passage time (FPT) density")
        println("  Eps floor: $eps_floor")
    end
    println("Data file: $data_file")
    println("Output file: $output_file")
    println("Parameter bounds:")
    
    for (name, bounds) in config.hard_bounds.dims
        scale = length(bounds) > 2 && bounds[3] == :log ? " (log scale)" : ""
        println("  $name: [$(bounds[1]), $(bounds[2])]$scale")
    end
    println("="^60)
    
    # Load data
    println("Loading data...")
    subject_trials = load_data_by_subject(data_file)
    println("Loaded data for $(length(subject_trials)) subjects")
    
    # Count trials per participant
    println("Counting trials per participant for BIC calculation...")
    trial_counts = count_trials_per_participant(data_file)
    
    # Run parallel fitting
    println("Starting parallel fitting...")
    pairs = collect(subject_trials)
    results = pmap(x -> fit_subject(x[1], x[2], model_name, likelihood_method; 
                                   kde_mode=kde_mode, bw_rule=bw_rule, J=J,
                                   min_sims=min_sims, min_matching=min_matching, max_sims=max_sims,
                                   eps_floor=eps_floor, lambda=lambda, optimizer=optimizer), pairs)
    
    # Collect and save results
    println("Collecting results...")
    
    # Get parameter names from model_configs
    config = get_model_config(model_name)
    param_names = config.param_names
    n_params = length(param_names)
    param_count = config.param_nums
    
    column_names = [:wid; Symbol.(param_names); :neglogl; :param_count; :n_trials; :bic; :optimizer]
    column_types = [String; fill(Float64, n_params); Float64; Int; Int; Float64; String]
    df = DataFrame([T[] for T in column_types], column_names)
    
    for (wid, θ, negll, _, optimizer) in results
        n_trials = get(trial_counts, wid, 0)
        bic = param_count * log(n_trials) + 2 * negll
        row_data = [wid; θ; negll; param_count; n_trials; bic; optimizer]
        push!(df, row_data)
    end
    
    CSV.write(output_file, df)
    println("Results saved to: $output_file")
    
    println("\nFitting Summary:")
    println("Successfully fitted $(length(results)) subjects")
    println("Mean negative log-likelihood: $(round(mean(df.neglogl), digits=2))")
    println("Std negative log-likelihood: $(round(std(df.neglogl), digits=2))")
    
    return df
end


function run_pda_fitting(model_name::String; kde_mode::Symbol=:gaussian, bw_rule::Symbol=:silverman, J::Int=500, 
                         min_sims::Int=1000, min_matching::Int=100, max_sims::Int=10000,
                         eps_floor::Float64=1e-16, lambda::Float64=1.0, optimizer::Symbol=:DE, kwargs...)
    return run_model_fitting(model_name; likelihood_method="pda", kde_mode=kde_mode, bw_rule=bw_rule, 
                            J=J, min_sims=min_sims, min_matching=min_matching, max_sims=max_sims,
                            eps_floor=eps_floor, lambda=lambda, optimizer=optimizer, kwargs...)
end

function run_ibs_fitting(model_name::String; optimizer::Symbol=:DE, kwargs...)
    return run_model_fitting(model_name; likelihood_method="ibs", kwargs...)
end

# run_pda_fitting("model6"; kde_mode=:gaussian, bw_rule=:silverman, J=1000, lambda=1.0, optimizer=:DE)