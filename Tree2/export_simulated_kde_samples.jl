# Export trial-specific simulated RT samples for Python plotting.
#
# This helper is intentionally thin: Python calls it, Julia loads the fitted
# parameters and participant trial, runs simulate_batch, filters to simulations
# matching the target choice pair, and writes rt1/rt2 samples to CSV.

include("model.jl")
include("data.jl")
include("pda.jl")
include("model_configs.jl")

using CSV, DataFrames

function _get_arg(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function _get_arg_int(flag::String, default::Int)
    return parse(Int, _get_arg(flag, string(default)))
end

const MODEL_NAME = _get_arg("--model", "model6")
const PARTICIPANT_ID = _get_arg("--participant", "w6eb2a0a")
const TRIAL_INDEX = _get_arg_int("--trial", 68)
const J = _get_arg_int("--samples", 1000)
const DATA_FILE = _get_arg("--data-file", joinpath(@__DIR__, "data", "Tree2_v3.json"))
const PDA_RESULTS_FILE = _get_arg("--params-file", joinpath(@__DIR__, "results", "pda", "model6_pda_BADS_20260125_211706.csv"))
const OUTPUT_FILE = _get_arg("--output", joinpath(@__DIR__, "figures", "simulated_kde_samples.csv"))

function main()
    config = get_model_config(MODEL_NAME)
    model_func = config.model_function

    trials_by_wid = load_data_by_subject(DATA_FILE)
    !haskey(trials_by_wid, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in $DATA_FILE")
    participant_trials = trials_by_wid[PARTICIPANT_ID]
    TRIAL_INDEX > length(participant_trials) && error("Trial index $TRIAL_INDEX exceeds available trials ($(length(participant_trials)))")
    target_trial = participant_trials[TRIAL_INDEX]

    pda_param_dict = load_fitted_parameters(PDA_RESULTS_FILE, MODEL_NAME)
    !haskey(pda_param_dict, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in $PDA_RESULTS_FILE")
    params = pda_param_dict[PARTICIPANT_ID]

    println("Simulating $J samples for participant=$PARTICIPANT_ID trial=$TRIAL_INDEX model=$MODEL_NAME")
    println("Target choice pair: ($(target_trial.choice1), $(target_trial.choice2))")
    println("Target RT: ($(target_trial.rt1), $(target_trial.rt2))")

    results = simulate_batch(model_func, params, target_trial.rewards, J)
    rows = NamedTuple[]
    for r in results
        if !r.timeout && r.choice1 == target_trial.choice1 && r.choice2 == target_trial.choice2
            push!(rows, (
                rt1 = Float64(r.rt1),
                rt2 = Float64(r.rt2),
                choice1 = Int(r.choice1),
                choice2 = Int(r.choice2),
                human_rt1 = Float64(target_trial.rt1),
                human_rt2 = Float64(target_trial.rt2),
                participant = PARTICIPANT_ID,
                trial_index = TRIAL_INDEX,
                model = MODEL_NAME,
            ))
        end
    end

    isempty(rows) && error("No non-timeout simulations matched the target choice pair.")

    outdir = dirname(OUTPUT_FILE)
    isdir(outdir) || mkpath(outdir)
    CSV.write(OUTPUT_FILE, DataFrame(rows))
    println("Exported $(length(rows)) matching simulated samples to $OUTPUT_FILE")
end

main()
