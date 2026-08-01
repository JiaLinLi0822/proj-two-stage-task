# fit_with_bads.jl  
# Parallel subject fitting using IBS and Bayesian Adaptive Direct Search (BADS) via Julia wrapper

using Distributed
addprocs(4)

@everywhere begin
    include("ibs.jl")
    include("model.jl")
    include("likelihood.jl")
    include("data.jl")
    include("bads.jl") 

    using JSON, DataFrames, CSV, Logging
    disable_logging(Logging.Warn)
end

@everywhere function fit_subject(wid, trials)

    lb  = [1e-10, 1e-10, 1e-3, 1e-3, 10.0, 10.0]      
    ub  = [1e-3,  1e-3,  2.0,  2.0,  10000.0, 10000.0]  
    plb = [1e-8,  1e-8,  1e-2, 1e-2, 50.0, 50.0]     
    pub = [1e-4,  1e-4,  1.0,  1.0,  8000.0, 8000.0] 
    x0  = [8e-5,  6e-5,  0.5, 0.8, 500.0, 500.0]

    println("Worker $(myid()): Starting BADS optimization for subject $wid")
    
    # Tracker
    eval_count = Ref(0)
    
    # Objective function
    function objective_function(θ)
        try
            eval_count[] += 1
            if eval_count[] % 100 == 0
                println("Worker $(myid()): Subject $wid - Evaluation $(eval_count[])")
            end
        
            model = Model(model1, θ)
    
            res = ibs_loglike(model, trials;
                              repeats  = 10,
                              max_iter = 1000,
                              ε        = 0.05,
                              rt_tol1  = 1000,
                              rt_tol2  = 1000,
                              min_multiplier = 0.8)
        
            neg_ll = res.neg_logp
    
            if !isfinite(neg_ll) || neg_ll < 0
                @error "Worker $(myid()): Bad negative log-likelihood estimate for $wid: $neg_ll"
            end
    
            return Float64(neg_ll)
    
        catch e
            @error "Worker $(myid()): Exception for $wid: $(e)"
            return 1e6
        end
    end
    
    try
        bads_result = optimize_bads(objective_function;
            x0 = x0,
            lower_bounds = lb,
            upper_bounds = ub, 
            plausible_lower_bounds = plb,
            plausible_upper_bounds = pub,
            max_fun_evals = 1000,
            uncertainty_handling = true
        )
        
        result_dict = get_result(bads_result)
        xopt = result_dict["x"]
        fopt = result_dict["fval"]
        
        println("Worker $(myid()): BADS completed for subject $wid ($(eval_count[]) evaluations)")
        println("Worker $(myid()): Subject $wid - Final θ = $xopt, negLL = $fopt")
        
        return wid, xopt, fopt
        
    catch e
        println("Worker $(myid()): BADS failed for subject $wid: $(typeof(e).name)")
        # If BADS fails, return the initial point and a large function value
        return wid, x0, 1e6
    end
end

# —— Main ——
# Load data using the proper data loading function
subject_trials = load_data_by_subject("data/Tree2_v3.json")

# parallel fit
pairs = collect(subject_trials)
results = pmap(x-> fit_subject(x[1], x[2]), pairs)

# collect and save
df = DataFrame(
    wid = String[], θ1 = Float64[], θ2 = Float64[], θ3 = Float64[],
    θ4 = Float64[], θ5 = Float64[], θ6 = Float64[], neglogl=Float64[]
)
for (wid, θ, negll) in results
    push!(df, (wid, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], negll))
end
CSV.write("julia/results_0708.csv", df)