# Script to plot participant-specific simulated RT samples as a 2D scatter
# and the corresponding joint KDE as a 3D surface.
#
# This mirrors plot_participant_kde.jl, but replaces the 2D heatmap KDE panel
# with a 3D KDE surface and adds a clean 2D scatter panel.

include("model.jl")
include("data.jl")
include("likelihood.jl")
include("pda.jl")
include("model_configs.jl")

using JSON, CSV, DataFrames, Statistics, Plots
using StatsBase: sample
using Random
using Printf

# ----------------------------- Global plot style -----------------------------
function setup_plot_style!()
    default(
        fontfamily = "Arial",
        guidefont  = font(8, "Arial"),
        tickfont   = font(7, "Arial"),
        legendfont = font(7, "Arial"),
        titlefont  = font(9, "Arial"),
        framestyle = :axes,
        grid       = false,
        dpi        = 300,
    )
end
setup_plot_style!()

# ----------------------------- Configuration --------------------------------
const USE_MANUAL_REWARDS = false
const MANUAL_REWARDS = [3.0, 4.0, -1.0, 0.0, -3.0, 0.0]
const MANUAL_PARAMS  = [9.71957139032952E-05, 0.0004551484937598540, 0.9308877667750590, 0.17884533984900000, 1177.4816254295100, 1479.0551306175400]

const PARTICIPANT_ID = "w6eb2a0a"
const TRIAL_INDEX    = 68
const J              = 1000
const MODEL_NAME     = "model6"
const IMMEDIATE_MS_THRESHOLD = 1.0

# KDE configuration
const KDE_MODE       = :gaussian    # :product or :gaussian
const BW_RULE        = :silverman   # :silverman or :scott (for :gaussian mode)
const LOG_RT         = true
const EPS_FLOOR      = 1e-16

# Plot in log(RT) coordinates by default.
const PLOT_LOG_SPACE = true
const MANUAL_RT1_RANGE = (2000, 10000)
const MANUAL_RT2_RANGE = (0, 6000)
const MANUAL_LOG_RT1_RANGE = 7.5, 9
const MANUAL_LOG_RT2_RANGE = 6.5, 9

const GRID_NX = 120
const GRID_NY = 120
const MAX_SCATTER_POINTS = 1000

println("="^80)
println("Participant / Synthetic 3D KDE Analysis")
println("="^80)
println("Mode: " * (USE_MANUAL_REWARDS ? "MANUAL_REWARDS (synthetic trial)" : "PARTICIPANT_DATA"))
println("Model: $MODEL_NAME")
println("Simulations per parameter set: $J")
println("KDE Mode: $KDE_MODE" * (KDE_MODE == :gaussian ? " (Gaussian KDE, rule: $BW_RULE)" : " (Product kernel)"))
println("Log RT for KDE fitting: $LOG_RT")
println("Plot Space: $(PLOT_LOG_SPACE ? "Log space" : "Original ms space")")

# ----------------------------- File paths -----------------------------------
const DATA_FILE        = joinpath(@__DIR__, "data", "Tree2_v3.json")
const PDA_RESULTS_FILE = joinpath(@__DIR__, "results", "pda", "model6_pda_BADS_20260125_211706.csv")

function analyze_parameter_set(param_name::String, params::Vector{Float64},
                               target_trial::Trial, model_func::Function)
    println("\n" * "-"^50)
    println("Analyzing: $param_name")
    println("Parameters: $params")
    println("-"^50)

    println("Running $J simulations...")
    results = simulate_batch(model_func, params, target_trial.rewards, J)
    isempty(results) && (println("No valid simulations generated"); return nothing)

    c1s = [r.choice1 for r in results if !r.timeout]
    rt1s = [r.rt1 for r in results if !r.timeout]
    c2s = [r.choice2 for r in results if !r.timeout]
    rt2s = [r.rt2 for r in results if !r.timeout]

    isempty(c1s) && (println("No valid simulations generated"); return nothing)
    println("Generated $(length(c1s)) valid samples")

    config = get_model_config(MODEL_NAME)
    T2_idx = findfirst(==("T2"), config.param_names)
    T2 = T2_idx !== nothing ? params[T2_idx] : 0.0
    n_immediate = T2_idx !== nothing ? count(rt2s .<= T2 + IMMEDIATE_MS_THRESHOLD) : 0
    pct_immediate = length(rt2s) > 0 ? round(100 * n_immediate / length(rt2s), digits=1) : 0.0
    println("Immediate termination (rt2 <= T2 + $(IMMEDIATE_MS_THRESHOLD) ms): $n_immediate / $(length(rt2s)) ($pct_immediate%)")

    tgt_c1, tgt_c2 = target_trial.choice1, target_trial.choice2
    idx = findall(i -> c1s[i] == tgt_c1 && c2s[i] == tgt_c2, eachindex(c1s))
    isempty(idx) && (println("No simulations matched target choice pair ($tgt_c1, $tgt_c2)"); return nothing)

    mrt1, mrt2 = rt1s[idx], rt2s[idx]
    println("Found $(length(idx)) simulations matching choice pair ($tgt_c1, $tgt_c2)")
    println("RT1 stats: mean=$(round(mean(mrt1), digits=1)), std=$(round(std(mrt1), digits=1))")
    println("RT2 stats: mean=$(round(mean(mrt2), digits=1)), std=$(round(std(mrt2), digits=1))")

    p1 = round(100 * mean(mrt1 .<= target_trial.rt1), digits=1)
    p2 = round(100 * mean(mrt2 .<= target_trial.rt2), digits=1)
    println("Participant RT1 ($(target_trial.rt1)) is at $(p1)th percentile")
    println("Participant RT2 ($(target_trial.rt2)) is at $(p2)th percentile")

    return Dict(
        :param_name => param_name,
        :matching_rt1s => mrt1,
        :matching_rt2s => mrt2,
        :rt1_percentile => p1,
        :rt2_percentile => p2,
        :n_total => length(c1s),
        :n_matching => length(idx),
    )
end

function compute_kde2d_grid(x::Vector{Float64}, y::Vector{Float64},
                            xgrid::Vector{Float64}, ygrid::Vector{Float64};
                            kde_mode::Symbol=:gaussian, bw_rule::Symbol=:silverman)
    @assert length(x) == length(y)
    n = length(x)
    n < 2 && return fill(0.0, length(ygrid), length(xgrid))

    kde_obj = if kde_mode == :product
        fit_kde2d_product(x, y; logRT=LOG_RT, eps_floor=EPS_FLOOR)
    elseif kde_mode == :gaussian || kde_mode == :full
        fit_kde2d_gaussian(x, y; logRT=LOG_RT, bw_rule=bw_rule, eps_floor=EPS_FLOOR)
    else
        error("Unknown kde_mode: $kde_mode (use :product or :gaussian)")
    end

    Z = Matrix{Float64}(undef, length(ygrid), length(xgrid))
    for (j, yy) in enumerate(ygrid)
        for (i, xx) in enumerate(xgrid)
            Z[j, i] = exp(logpdf(kde_obj, xx, yy))
        end
    end
    return Z
end

function _plot_ranges(mrt1s, mrt2s, target_trial::Trial)
    if PLOT_LOG_SPACE
        if MANUAL_LOG_RT1_RANGE !== nothing
            plot_xr = MANUAL_LOG_RT1_RANGE
            xr = (exp(plot_xr[1]), exp(plot_xr[2]))
        else
            xr_min = min(minimum(mrt1s), target_trial.rt1)
            xr_max = max(maximum(mrt1s), target_trial.rt1)
            pad = (xr_max - xr_min) * 0.1
            xr = (max(1.0, xr_min - pad), min(10000.0, xr_max + pad))
            plot_xr = (log(xr[1]), log(xr[2]))
        end

        if MANUAL_LOG_RT2_RANGE !== nothing
            plot_yr = MANUAL_LOG_RT2_RANGE
            yr = (exp(plot_yr[1]), exp(plot_yr[2]))
        else
            yr_min = min(minimum(mrt2s), target_trial.rt2)
            yr_max = max(maximum(mrt2s), target_trial.rt2)
            pad = (yr_max - yr_min) * 0.1
            yr = (max(1.0, yr_min - pad), min(10000.0, yr_max + pad))
            plot_yr = (log(yr[1]), log(yr[2]))
        end
        return xr, yr, plot_xr, plot_yr
    end

    xr = MANUAL_RT1_RANGE !== nothing ? MANUAL_RT1_RANGE : begin
        xr_min = min(minimum(mrt1s), target_trial.rt1)
        xr_max = max(maximum(mrt1s), target_trial.rt1)
        pad = (xr_max - xr_min) * 0.1
        (max(100.0, xr_min - pad), min(10000.0, xr_max + pad))
    end
    yr = MANUAL_RT2_RANGE !== nothing ? MANUAL_RT2_RANGE : begin
        yr_min = min(minimum(mrt2s), target_trial.rt2)
        yr_max = max(maximum(mrt2s), target_trial.rt2)
        pad = (yr_max - yr_min) * 0.1
        (max(0.0, yr_min - pad), min(10000.0, yr_max + pad))
    end
    return xr, yr, xr, yr
end

function _subsample_indices(n::Int, max_n::Int)
    n <= max_n && return collect(1:n)
    return sort!(sample(1:n, max_n; replace=false))
end

function create_scatter_and_3d_kde_plots(results::Vector, target_trial::Trial)
    println("\n" * "="^50)
    println("Creating 2D scatter + 3D KDE surface plots...")
    println("="^50)

    figures_dir = joinpath(@__DIR__, "figures")
    isdir(figures_dir) || mkpath(figures_dir)

    for res in results
        (res === nothing || res[:n_matching] < 10) && continue

        name  = res[:param_name]
        mrt1s = res[:matching_rt1s]
        mrt2s = res[:matching_rt2s]
        xr, yr, plot_xr, plot_yr = _plot_ranges(mrt1s, mrt2s, target_trial)

        if PLOT_LOG_SPACE
            plot_mrt1s = log.(max.(1e-10, mrt1s))
            plot_mrt2s = log.(max.(1e-10, mrt2s))
            plot_trial_rt1 = log(max(1e-10, target_trial.rt1))
            plot_trial_rt2 = log(max(1e-10, target_trial.rt2))
        else
            plot_mrt1s = mrt1s
            plot_mrt2s = mrt2s
            plot_trial_rt1 = target_trial.rt1
            plot_trial_rt2 = target_trial.rt2
        end

        xgrid = collect(range(plot_xr[1], plot_xr[2], length=GRID_NX))
        ygrid = collect(range(plot_yr[1], plot_yr[2], length=GRID_NY))
        Z = compute_kde2d_grid(plot_mrt1s, plot_mrt2s, xgrid, ygrid;
                               kde_mode=KDE_MODE, bw_rule=BW_RULE)

        xlabel_text = PLOT_LOG_SPACE ? "log(RT1)" : "RT1 (ms)"
        ylabel_text = PLOT_LOG_SPACE ? "log(RT2)" : "RT2 (ms)"

        idx = _subsample_indices(length(plot_mrt1s), MAX_SCATTER_POINTS)
        p_scatter = scatter(
            plot_mrt1s[idx], plot_mrt2s[idx];
            xlabel=xlabel_text,
            ylabel=ylabel_text,
            xlim=plot_xr,
            ylim=plot_yr,
            label="Simulated samples",
            color=RGBA(0.42, 0.33, 0.82, 0.35),
            markerstrokewidth=0,
            markersize=3.6,
            legend=:topright,
            title="2D samples",
            margin=4Plots.mm,
        )
        scatter!(p_scatter, [plot_trial_rt1], [plot_trial_rt2];
                 markershape=:star5,
                 markersize=9,
                 color=:red,
                 markerstrokecolor=:white,
                 markerstrokewidth=1.2,
                 label="Human RT")

        zmax = maximum(Z)
        zticks = ([0.0, zmax], ["0", @sprintf("%.2e", zmax)])
        kde_title = KDE_MODE == :gaussian ? "3D Gaussian KDE" : "3D Product KDE"
        p_surface = surface(
            xgrid, ygrid, permutedims(Z);
            xlabel=xlabel_text,
            ylabel=ylabel_text,
            zlabel="KDE density",
            xlim=plot_xr,
            ylim=plot_yr,
            zlim=(0.0, zmax * 1.05),
            color=:viridis,
            linecolor=RGBA(0.0, 0.0, 0.0, 0.22),
            linewidth=0.25,
            colorbar=false,
            legend=false,
            camera=(35, 28),
            title=kde_title,
            margin=4Plots.mm,
            zticks=zticks,
            grid=true,
            gridalpha=0.25,
        )
        scatter!(p_surface, [plot_trial_rt1], [plot_trial_rt2], [zmax * 1.02];
                 markershape=:star5,
                 markersize=5,
                 color=:red,
                 markerstrokecolor=:white,
                 markerstrokewidth=0.8,
                 label="")

        fig = plot(
            p_scatter,
            p_surface;
            layout=(1, 2),
            size=(860, 360),
            dpi=300,
            margin=2Plots.mm,
        )

        base_id = USE_MANUAL_REWARDS ? "synthetic_rewards" : PARTICIPANT_ID
        base_trial = USE_MANUAL_REWARDS ? "trial_auto" : string(TRIAL_INDEX)
        fname = "participant_$(base_id)_trial_$(base_trial)_scatter_3dkde_$(replace(name, ' '=>'_')).png"
        full_path = joinpath(figures_dir, fname)
        savefig(fig, full_path)
        println("Saved 2D scatter + 3D KDE plot: $full_path")
    end
end

# ----------------------------- Main -----------------------------------------
try
    config = get_model_config(MODEL_NAME)
    model_func = config.model_function

    target_trial = Trial("init", Float64[], 0, 0, 0.0, 0.0, Float64[])
    all_results = Any[]

    if USE_MANUAL_REWARDS
        println("\nUsing MANUAL_REWARDS mode (synthetic trial)...")
        @assert length(MANUAL_REWARDS) == 6 "MANUAL_REWARDS must have length 6"
        @assert length(MANUAL_PARAMS) == length(config.param_names) "MANUAL_PARAMS length must match model parameter count"

        result = model_func(MANUAL_PARAMS, MANUAL_REWARDS)
        result.timeout && error("Simulation with MANUAL_REWARDS and MANUAL_PARAMS timed out; cannot build target trial.")

        target_trial = Trial(
            "synthetic_manual",
            MANUAL_REWARDS,
            result.choice1,
            result.choice2,
            result.rt1,
            result.rt2,
            MANUAL_REWARDS[3:6],
        )
        println("Synthetic target trial:")
        println("  Choice  = ($(target_trial.choice1), $(target_trial.choice2))")
        println("  RTs     = ($(target_trial.rt1), $(target_trial.rt2))")
        println("  Rewards = $(target_trial.rewards)")

        push!(all_results, analyze_parameter_set("Manual Params", MANUAL_PARAMS, target_trial, model_func))
    else
        println("\nLoading participant data...")
        trials_by_wid = load_data_by_subject(DATA_FILE)
        !haskey(trials_by_wid, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in data")
        participant_trials = trials_by_wid[PARTICIPANT_ID]
        TRIAL_INDEX > length(participant_trials) && error("Trial index $TRIAL_INDEX exceeds available trials ($(length(participant_trials)))")

        target_trial = participant_trials[TRIAL_INDEX]
        println("Target trial: Choice ($(target_trial.choice1), $(target_trial.choice2)), RT ($(target_trial.rt1), $(target_trial.rt2))")
        println("Rewards: $(target_trial.rewards)")

        println("\nLoading PDA fitted parameters...")
        pda_param_dict = load_fitted_parameters(PDA_RESULTS_FILE, MODEL_NAME)
        !haskey(pda_param_dict, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in PDA results")
        pda_params = pda_param_dict[PARTICIPANT_ID]
        println("PDA parameters loaded: $pda_params")

        push!(all_results, analyze_parameter_set("PDA Fitted", pda_params, target_trial, model_func))
    end

    create_scatter_and_3d_kde_plots(all_results, target_trial)

    println("\n" * "="^80)
    println(USE_MANUAL_REWARDS ? "Analysis Summary for Synthetic Manual Trial" : "Analysis Summary for Participant $PARTICIPANT_ID, Trial $TRIAL_INDEX")
    println("="^80)
    for res in all_results
        res === nothing && continue
        println("$(res[:param_name]):")
        println("  - Generated $(res[:n_matching])/$(res[:n_total]) matching simulations")
        println("  - Target RT1 at $(res[:rt1_percentile])th percentile")
        println("  - Target RT2 at $(res[:rt2_percentile])th percentile")
    end
catch e
    println("Analysis failed: $e")
    println("\nTroubleshooting:")
    println("1. If USE_MANUAL_REWARDS = false, make sure data file exists: $DATA_FILE")
    println("2. If using participant mode, ensure results file exists and has participant ID: $PDA_RESULTS_FILE")
    println("3. Check that PARTICIPANT_ID and TRIAL_INDEX are valid")
    println("4. If USE_MANUAL_REWARDS = true, verify MANUAL_REWARDS and MANUAL_PARAMS shapes")
    rethrow(e)
end
