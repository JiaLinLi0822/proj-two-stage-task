# Script to analyze participant-specific KDE using fitted parameters
# Compares IBS vs PDA fitted parameters and random parameters

include("model.jl")
include("data.jl")
include("likelihood.jl")
include("pda.jl")
include("model_configs.jl")

using JSON, CSV, DataFrames, Statistics, Plots
using Random
using Printf
Random.seed!(20250830)

# ----------------------------- Global plot style -----------------------------
function setup_plot_style!()
    default(
        fontfamily = "Arial",
        guidefont  = font(7, "Arial"),
        tickfont   = font(7, "Arial"),
        legendfont = font(7, "Arial"),
        titlefont  = font(8, "Arial"),
        framestyle = :axes,
        grid       = false,
        dpi        = 300, 
    )
end
setup_plot_style!()

# ----------------------------- Configuration --------------------------------
const PARTICIPANT_ID = "wdaebe9a"
const TRIAL_INDEX    = 82
const J              = 1000
const MODEL_NAME     = "model6"

# KDE Configuration
const KDE_MODE       = :gaussian    # :product or :gaussian
const BW_RULE        = :silverman   # :silverman or :scott (for :gaussian mode)
const LOG_RT         = true
const EPS_FLOOR      = 1e-16

# Plot Configuration - Control log transformation for visualization
const PLOT_LOG_SPACE = false         # true: plot in log space, false: plot in original space

println("="^80)
println("Participant-Specific KDE Analysis")
println("="^80)
println("Participant: $PARTICIPANT_ID")
println("Trial: $TRIAL_INDEX")
println("Model: $MODEL_NAME")
println("Simulations: $J")
println("KDE Mode: $KDE_MODE" * (KDE_MODE == :gaussian ? " (Gaussian KDE, rule: $BW_RULE)" : " (Product kernel)"))
println("Log RT: $LOG_RT")
println("EPS Floor: $EPS_FLOOR")
println("Plot Space: $(PLOT_LOG_SPACE ? "Log space" : "Original space")")

# ----------------------------- File paths -----------------------------------
const DATA_FILE       = "Tree2/data/Tree2_v3.json"
const PDA_RESULTS_FILE = "Tree2/results/pda/model6_pda_BADS_20251003_153755.csv"
const IBS_RESULTS_FILE = "Tree2/results/ibs/model6_ibs_20250711_005050.csv"

# ----------------------------- Utilities ------------------------------------
_pad(a, b; include=0.0, floor=100.0, pct=0.2) = (max(floor, 0.8*min(a, include)), 1.2*max(b, include))

function _safe_choice_reward(trial::Trial)
    c1r = trial.choice1 == 1 ? trial.rewards[1] : trial.rewards[2]
    c2r = trial.choice1 == 1 ?
          (trial.choice2 == 1 ? trial.rewards[3] : trial.rewards[4]) :
          (trial.choice2 == 1 ? trial.rewards[5] : trial.rewards[6])
    return c1r, c2r
end

# ----------------------------- Analysis helpers -----------------------------
function analyze_parameter_set(param_name::String, params::Vector{Float64},
                               target_trial::Trial, model_func::Function)
    println("\n" * "-"^50)
    println("Analyzing: $param_name")
    println("Parameters: $params")
    println("-"^50)

    println("Running $J simulations...")
    results = simulate_batch(model_func, params, target_trial.rewards, J)
    isempty(results) && (println("❌ No valid simulations generated"); return nothing)

    # Extract data from results
    c1s = [r.choice1 for r in results if !r.timeout]
    rt1s = [r.rt1 for r in results if !r.timeout]
    c2s = [r.choice2 for r in results if !r.timeout]
    rt2s = [r.rt2 for r in results if !r.timeout]
    
    isempty(c1s) && (println("❌ No valid simulations generated"); return nothing)

    println("Generated $(length(c1s)) valid samples")
    choice1_dist = Dict(c => count(==(c), c1s) for c in unique(c1s))
    choice2_dist = Dict(c => count(==(c), c2s) for c in unique(c2s))
    println("Overall Choice1 distribution: $choice1_dist")
    println("Overall Choice2 distribution: $choice2_dist")

    tgt_c1, tgt_c2 = target_trial.choice1, target_trial.choice2
    idx = findall(i -> c1s[i] == tgt_c1 && c2s[i] == tgt_c2, eachindex(c1s))
    isempty(idx) && (println("❌ No simulations matched target choice pair ($tgt_c1, $tgt_c2)"); return nothing)

    mrt1, mrt2 = rt1s[idx], rt2s[idx]
    println("Found $(length(idx)) simulations matching choice pair ($tgt_c1, $tgt_c2)")
    println("RT1 stats: mean=$(round(mean(mrt1), digits=1)), std=$(round(std(mrt1), digits=1))")
    println("RT2 stats: mean=$(round(mean(mrt2), digits=1)), std=$(round(std(mrt2), digits=1))")

    p1 = round(100 * mean(mrt1 .<= target_trial.rt1), digits=1)
    p2 = round(100 * mean(mrt2 .<= target_trial.rt2), digits=1)
    println("Participant RT1 ($(target_trial.rt1)) is at $(p1)th percentile")
    println("Participant RT2 ($(target_trial.rt2)) is at $(p2)th percentile")

    return Dict(
        :param_name => param_name, :params => params,
        :all_rt1s => rt1s, :all_rt2s => rt2s,
        :matching_rt1s => mrt1, :matching_rt2s => mrt2,
        :rt1_percentile => p1, :rt2_percentile => p2,
        :n_total => length(c1s), :n_matching => length(idx)
    )
end

# ----------------------------- KDE (2D) --------------------------------

function compute_kde2d_grid(x::Vector{Float64}, y::Vector{Float64},
                            xgrid::Vector{Float64}, ygrid::Vector{Float64};
                            kde_mode::Symbol=:gaussian, bw_rule::Symbol=:silverman)
    @assert length(x) == length(y)
    n = length(x)
    n < 2 && return fill(0.0, length(ygrid), length(xgrid))

    # Fit KDE object using the latest pda.jl implementations
    kde_obj = if kde_mode == :product
        fit_kde2d_product(x, y; logRT=LOG_RT, eps_floor=EPS_FLOOR)
    elseif kde_mode == :gaussian || kde_mode == :full
        # :full kept as alias for backward compatibility
        fit_kde2d_gaussian(x, y; logRT=LOG_RT, bw_rule=bw_rule, eps_floor=EPS_FLOOR)
    else
        error("Unknown kde_mode: $kde_mode (use :product or :gaussian)")
    end

    # Evaluate on the grid
    Z = Matrix{Float64}(undef, length(ygrid), length(xgrid))
    for (j, yy) in enumerate(ygrid)
        for (i, xx) in enumerate(xgrid)
            Z[j, i] = exp(logpdf(kde_obj, xx, yy))
        end
    end
    return Z
end

# ----------------------------- Plot helpers ---------------------------------
function _plot_tree(trial::Trial)
    root_x, root_y = 0.5, 0.8
    left_x, left_y = 0.25, 0.4
    right_x, right_y = 0.75, 0.4
    leaf = ((0.125,0.1),(0.375,0.1),(0.625,0.1),(0.875,0.1))

    plt = plot(legend=false, framestyle=:none, xticks=false, yticks=false, size=(300,200), margin=0Plots.mm)
    scatter!(plt, [root_x], [root_y]; markershape=:circle, ms=8, color=:black, markerstrokewidth=0)
    scatter!(plt, [left_x, right_x], [left_y, right_y]; markershape=:circle, ms=6, color=:black, markerstrokewidth=0)
    scatter!(plt, [l[1] for l in leaf], [l[2] for l in leaf]; markershape=:circle, ms=4, color=:black, markerstrokewidth=0)

    plot!(plt, [root_x, left_x],  [root_y, left_y];  color=:black, linewidth=1, label="")
    plot!(plt, [root_x, right_x], [root_y, right_y]; color=:black, linewidth=1, label="")
    plot!(plt, [left_x,  leaf[1][1]],  [left_y,  leaf[1][2]];  color=:black, linewidth=1, label="")
    plot!(plt, [left_x,  leaf[2][1]],  [left_y,  leaf[2][2]];  color=:black, linewidth=1, label="")
    plot!(plt, [right_x, leaf[3][1]],  [right_y, leaf[3][2]];  color=:black, linewidth=1, label="")
    plot!(plt, [right_x, leaf[4][1]],  [right_y, leaf[4][2]];  color=:black, linewidth=1, label="")

    annotate!(plt, left_x  - 0.05, left_y,  text("$(trial.rewards[1])", 8, :right))
    annotate!(plt, right_x - 0.05, right_y, text("$(trial.rewards[2])", 8, :right))
    annotate!(plt, leaf[1][1] - 0.05, leaf[1][2], text("$(trial.rewards[3])", 8, :right))
    annotate!(plt, leaf[2][1] - 0.05, leaf[2][2], text("$(trial.rewards[4])", 8, :right))
    annotate!(plt, leaf[3][1] - 0.05, leaf[3][2], text("$(trial.rewards[5])", 8, :right))
    annotate!(plt, leaf[4][1] - 0.05, leaf[4][2], text("$(trial.rewards[6])", 8, :right))

    if trial.choice1 == 1
        plot!(plt, [root_x, left_x], [root_y, left_y]; color=:red, linewidth=3, label="")
        if trial.choice2 == 1
            plot!(plt, [left_x, leaf[1][1]], [left_y, leaf[1][2]]; color=:red, linewidth=3, label="")
            scatter!(plt, [leaf[1][1]], [leaf[1][2]]; markershape=:circle, ms=6, color=:red, markerstrokewidth=0)
        else
            plot!(plt, [left_x, leaf[2][1]], [left_y, leaf[2][2]]; color=:red, linewidth=3, label="")
            scatter!(plt, [leaf[2][1]], [leaf[2][2]]; markershape=:circle, ms=6, color=:red, markerstrokewidth=0)
        end
    else
        plot!(plt, [root_x, right_x], [root_y, right_y]; color=:red, linewidth=3, label="")
        if trial.choice2 == 1
            plot!(plt, [right_x, leaf[3][1]], [right_y, leaf[3][2]]; color=:red, linewidth=3, label="")
            scatter!(plt, [leaf[3][1]], [leaf[3][2]]; markershape=:circle, ms=6, color=:red, markerstrokewidth=0)
        else
            plot!(plt, [right_x, leaf[4][1]], [right_y, leaf[4][2]]; color=:red, linewidth=3, label="")
            scatter!(plt, [leaf[4][1]], [leaf[4][2]]; markershape=:circle, ms=6, color=:red, markerstrokewidth=0)
        end
    end
    # annotate!(tree_plot, x, y, text("$(label)", 8, "Arial"))
    plt
end

function _marginal_plots(mrt1s, mrt2s, xr, yr, trial::Trial, Z, xgrid, ygrid)
    # Transform data for plotting based on PLOT_LOG_SPACE setting
    plot_rt1s = PLOT_LOG_SPACE ? log.(mrt1s) : mrt1s
    plot_rt2s = PLOT_LOG_SPACE ? log.(mrt2s) : mrt2s
    plot_trial_rt1 = PLOT_LOG_SPACE ? log(trial.rt1) : trial.rt1
    plot_trial_rt2 = PLOT_LOG_SPACE ? log(trial.rt2) : trial.rt2
    plot_xr = PLOT_LOG_SPACE ? (log(xr[1]), log(xr[2])) : xr
    plot_yr = PLOT_LOG_SPACE ? (log(yr[1]), log(yr[2])) : yr
    
    # Compute marginal distributions from 2D KDE
    rt1_marginal = vec(sum(Z, dims=1))  # Sum over y dimension
    rt2_marginal = vec(sum(Z, dims=2))  # Sum over x dimension
    
    # Normalize marginals to be proper density functions
    dx = xgrid[2] - xgrid[1]
    dy = ygrid[2] - ygrid[1]
    rt1_marginal ./= sum(rt1_marginal) * dx
    rt2_marginal ./= sum(rt2_marginal) * dy
    
    maxden = max(maximum(rt1_marginal), maximum(rt2_marginal))
    dens_lim = (0, maxden * 1.1)

    # Top (RT1)
    rt1_y = (0, maxden * 1.2)
    xlabel_text = PLOT_LOG_SPACE ? "log(RT1)" : "RT1 (ms)"
    p_top = plot(xlim=plot_xr, ylim=rt1_y, xlabel="", ylabel="Density",
                 legend=:topright, title="RT1 marginal", yformatter=:scientific)
    # Plot histogram using bar plot
    histogram!(p_top, plot_rt1s; bins=20, normalize=:pdf, alpha=0.7, color=:lightblue, label="Histogram")
    plot!(p_top, xgrid, rt1_marginal; color=:darkblue, linewidth=2, label="2D KDE marginal")
    vline!(p_top, [plot_trial_rt1]; color=:red, linewidth=2, linestyle=:solid, label="Participant RT1")
    vline!(p_top, [mean(plot_rt1s)]; color=:blue, linewidth=2, linestyle=:dash, label="Simulation mean")

    # Right (RT2)
    ylabel_text = PLOT_LOG_SPACE ? "log(RT2)" : "RT2 (ms)"
    p_right = plot(xlim=dens_lim, ylim=plot_yr, xlabel="Density", ylabel="",
                   legend=:topright, title="RT2 marginal", xrotation=270, xformatter=:scientific)
    # Plot histogram using bar plot
    histogram!(p_right, plot_rt2s; bins=20, normalize=:pdf, orientation=:h, alpha=0.7, color=:lightgreen, label="Histogram")
    plot!(p_right, rt2_marginal, ygrid; color=:darkgreen, linewidth=2, label="2D KDE marginal")
    hline!(p_right, [plot_trial_rt2]; color=:red, linewidth=2, linestyle=:solid, label="Participant RT2")
    hline!(p_right, [mean(plot_rt2s)]; color=:blue, linewidth=2, linestyle=:dash, label="Simulation mean")

    return p_top, p_right
end

# ----------------------------- Joint 2D plotting ----------------------------
function create_joint_kde_plots(results::Vector, target_trial::Trial)
    println("\n" * "="^50)
    println("Creating JOINT 2D KDE plots (with marginal histograms)...")
    println("="^50)

    try
        for res in results
            (res === nothing || res[:n_matching] < 10) && continue

            name     = res[:param_name]
            mrt1s    = res[:matching_rt1s]
            mrt2s    = res[:matching_rt2s]

            xr = _pad(minimum(mrt1s), maximum(mrt1s); include=target_trial.rt1)
            yr = _pad(minimum(mrt2s), maximum(mrt2s); include=target_trial.rt2)
            nx, ny = 120, 120
            
            # Transform data and grid for plotting based on PLOT_LOG_SPACE setting
            if PLOT_LOG_SPACE
                plot_mrt1s = log.(mrt1s)
                plot_mrt2s = log.(mrt2s)
                plot_xr = (log(xr[1]), log(xr[2]))
                plot_yr = (log(yr[1]), log(yr[2]))
                xgrid = collect(range(plot_xr[1], plot_xr[2], length=nx))
                ygrid = collect(range(plot_yr[1], plot_yr[2], length=ny))
                Z = compute_kde2d_grid(plot_mrt1s, plot_mrt2s, xgrid, ygrid; kde_mode=KDE_MODE, bw_rule=BW_RULE)
            else
                plot_mrt1s = mrt1s
                plot_mrt2s = mrt2s
                plot_xr = xr
                plot_yr = yr
                xgrid = collect(range(xr[1], xr[2], length=nx))
                ygrid = collect(range(yr[1], yr[2], length=ny))
                Z = compute_kde2d_grid(plot_mrt1s, plot_mrt2s, xgrid, ygrid; kde_mode=KDE_MODE, bw_rule=BW_RULE)
            end

            # ---- colorbar ticks in scientific notation ----
            zmin, zmax = extrema(Z)
            if zmax ≈ zmin
                zmax = zmin + eps(zmin)
            end
            cb_ticks  = collect(range(zmin, zmax, length=7))
            cb_labels = [@sprintf("%.1e", t) for t in cb_ticks]

            # Set axis labels based on plot space
            xlabel_text = PLOT_LOG_SPACE ? "log(RT1)" : "RT1 (ms)"
            ylabel_text = PLOT_LOG_SPACE ? "log(RT2)" : "RT2 (ms)"
            
            p_center = contourf(
                xgrid, ygrid, Z;
                fill=true, c=:viridis, levels=18,
                colorbar=true,
                colorbar_ticks=(cb_ticks, cb_labels), 
                xlabel=xlabel_text, ylabel=ylabel_text,
                xlim=plot_xr, ylim=plot_yr,
                title="2D KDE\n$name · Participant: $PARTICIPANT_ID · Trial: $TRIAL_INDEX\n($(length(mrt1s)) matching choices)",
                legend=:topright
            )

            if PLOT_LOG_SPACE
                scatter!(p_center, log.(mrt1s), log.(mrt2s);
                         markershape=:x, ms=3, ma=0.3, color=:black, label="samples")
            else
                scatter!(p_center, mrt1s, mrt2s;
                         markershape=:x, ms=3, ma=0.3, color=:black, label="samples")
            end

            if PLOT_LOG_SPACE
                scatter!(p_center, [log(target_trial.rt1)], [log(target_trial.rt2)];
                         markershape=:star5, ms=5, color=:red, label="participant RT")
            else
                scatter!(p_center, [target_trial.rt1], [target_trial.rt2];
                         markershape=:star5, ms=5, color=:red, label="participant RT")
            end

            p_top, p_right = _marginal_plots(mrt1s, mrt2s, xr, yr, target_trial, Z, xgrid, ygrid)
            tree_plot = _plot_tree(target_trial)

            layout_tpl = Plots.grid(2, 2; heights=[0.22, 0.78], widths=[0.78, 0.22])
            fig = plot(p_top, tree_plot, p_center, p_right; layout=layout_tpl, size=(900, 750), margin=5Plots.mm)

            # Create figures directory if it doesn't exist
            figures_dir = "Tree2/figures"
            if !isdir(figures_dir)
                mkpath(figures_dir)
            end
            
            fname = "participant_$(PARTICIPANT_ID)_trial_$(TRIAL_INDEX)_joint2d_$(replace(name, ' '=>'_')).png"
            full_path = joinpath(figures_dir, fname)
            savefig(fig, full_path)
            println("  ✅ Saved: $full_path")
        end
    catch e
        println("❌ Joint plotting failed: $e")
        println("Note: Make sure Plots.jl is installed: using Pkg; Pkg.add(\"Plots\")")
    end
end

# ----------------------------- Main -----------------------------------------
try
    println("\n📖 Loading participant data...")
    trials_by_wid = load_data_by_subject(DATA_FILE)
    !haskey(trials_by_wid, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in data")
    participant_trials = trials_by_wid[PARTICIPANT_ID]
    TRIAL_INDEX > length(participant_trials) && error("Trial index $TRIAL_INDEX exceeds available trials ($(length(participant_trials)))")

    target_trial = participant_trials[TRIAL_INDEX]
    println("Target trial: Choice ($(target_trial.choice1), $(target_trial.choice2)), RT ($(target_trial.rt1), $(target_trial.rt2))")
    println("Rewards: $(target_trial.rewards)")

    config = get_model_config(MODEL_NAME)
    model_func = config.model_function

    all_results = Any[]

    println("\n📊 Loading PDA fitted parameters...")
    try
        pda_param_dict = load_fitted_parameters(PDA_RESULTS_FILE, MODEL_NAME)
        !haskey(pda_param_dict, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in PDA results")
        pda_params = pda_param_dict[PARTICIPANT_ID]
        println("PDA parameters loaded: $pda_params")
        push!(all_results, analyze_parameter_set("PDA Fitted", pda_params, target_trial, model_func))
    catch e
        println("❌ Could not load PDA parameters: $e"); push!(all_results, nothing)
    end

    println("\n📊 Loading IBS fitted parameters...")
    try
        ibs_param_dict = load_fitted_parameters(IBS_RESULTS_FILE, MODEL_NAME)
        !haskey(ibs_param_dict, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in IBS results")
        ibs_params = ibs_param_dict[PARTICIPANT_ID]
        println("IBS parameters loaded: $ibs_params")
        push!(all_results, analyze_parameter_set("IBS Fitted", ibs_params, target_trial, model_func))
    catch e
        println("❌ Could not load IBS parameters: $e"); push!(all_results, nothing)
    end

    println("\n🎲 Using random parameters...")
    random_params = MODEL_NAME == "model1" ? [5e-5, 9e-5, 0.4, 1.1, 700.0, 350.0] :
                   MODEL_NAME == "model5" ? [6e-5, 0.4, 1.0, 600.0, 400.0] :
                   MODEL_NAME == "model6" ? [8e-5, 6e-5, 0.4, 1.1, 700.0, 350.0] :
                   MODEL_NAME == "model11" ? [6e-5, 0.7, 0.8, 1.5, 600.0, 400.0] :
                   error("Random params not defined for $MODEL_NAME")
    push!(all_results, analyze_parameter_set("Random", random_params, target_trial, model_func))

    # create_kde_plots(all_results, target_trial) # optional
    create_joint_kde_plots(all_results, target_trial)

    println("\n" * "="^80)
    println("Analysis Summary for Participant $PARTICIPANT_ID, Trial $TRIAL_INDEX")
    println("="^80)
    for res in all_results
        res === nothing && continue
        println("$(res[:param_name]):")
        println("  - Generated $(res[:n_matching])/$(res[:n_total]) matching simulations")
        println("  - Participant RT1 at $(res[:rt1_percentile])th percentile")
        println("  - Participant RT2 at $(res[:rt2_percentile])th percentile")
    end

    println("\n💡 Interpretation:")
    println("- Percentiles near 50% indicate participant RT is typical for that parameter set")
    println("- Very high/low percentiles suggest parameter set doesn't capture participant behavior well")
    println("- Compare PDA vs IBS fitted parameters to see which better captures the participant")

catch e
    println("❌ Analysis failed: $e")
    println("\n🔧 Troubleshooting:")
    println("1. Make sure data file exists: $DATA_FILE")
    println("2. Make sure results files exist and have correct participant IDs")
    println("3. Check that PARTICIPANT_ID and TRIAL_INDEX are valid")
    println("4. Verify model parameter structure matches CSV columns")
end