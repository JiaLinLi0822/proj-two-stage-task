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
# Random.seed!(20260130)

# ----------------------------- Global plot style -----------------------------
function setup_plot_style!()
    default(
        fontfamily = "Arial",
        guidefont  = font(7, "Arial"),
        tickfont   = font(8, "Arial"),
        legendfont = font(7, "Arial"),
        titlefont  = font(10, "Arial"),
        framestyle = :axes,
        grid       = false,
        dpi        = 500, 
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

# KDE Configuration
const KDE_MODE       = :gaussian    # :product or :gaussian
const BW_RULE        = :silverman   # :silverman or :scott (for :gaussian mode)
const LOG_RT         = true
const EPS_FLOOR      = 1e-16

const PLOT_LOG_SPACE = false

const MANUAL_RT1_RANGE = (2000, 10000)    
const MANUAL_RT2_RANGE = (0, 6000)     
const MANUAL_LOG_RT1_RANGE = log(500), log(10000)
const MANUAL_LOG_RT2_RANGE = log(100), log(10000) 

const SIZE_RT1_MARGINAL = (300, 140)
const SIZE_RT2_MARGINAL = (160, 360)
const SIZE_2D_KDE       = (400, 300)
const SIZE_TREE         = (300, 300)

println("="^80)
println("Participant / Synthetic KDE Analysis")
println("="^80)
println("Mode: " * (USE_MANUAL_REWARDS ? "MANUAL_REWARDS (synthetic trial)" : "PARTICIPANT_DATA"))
println("Model: $MODEL_NAME")
println("Simulations per parameter set: $J")
println("KDE Mode: $KDE_MODE" * (KDE_MODE == :gaussian ? " (Gaussian KDE, rule: $BW_RULE)" : " (Product kernel)"))
println("Log RT: $LOG_RT")
println("EPS Floor: $EPS_FLOOR")
println("Plot Space: $(PLOT_LOG_SPACE ? "Log space" : "Original space")")
if USE_MANUAL_REWARDS
    println("Manual rewards: $MANUAL_REWARDS")
    println("Manual params:  $MANUAL_PARAMS")
else
    println("Participant: $PARTICIPANT_ID")
    println("Trial index: $TRIAL_INDEX")
end

# ----------------------------- File paths -----------------------------------
const DATA_FILE        = joinpath(@__DIR__, "data", "Tree2_v3.json")
const PDA_RESULTS_FILE = joinpath(@__DIR__, "results", "pda", "model6_pda_BADS_20260125_211706.csv")
const IBS_RESULTS_FILE = joinpath(@__DIR__, "results", "ibs", "model6_ibs_20250711_005050.csv")

_pad(a, b; include=0.0, floor=100.0, pct=0.2) = (max(floor, 0.8*min(a, include)), 1.2*max(b, include))

function _safe_choice_reward(trial::Trial)
    c1r = trial.choice1 == 1 ? trial.rewards[1] : trial.rewards[2]
    c2r = trial.choice1 == 1 ?
          (trial.choice2 == 1 ? trial.rewards[3] : trial.rewards[4]) :
          (trial.choice2 == 1 ? trial.rewards[5] : trial.rewards[6])
    return c1r, c2r
end


function analyze_parameter_set(param_name::String, params::Vector{Float64},
                               target_trial::Trial, model_func::Function)
    println("\n" * "-"^50)
    println("Analyzing: $param_name")
    println("Parameters: $params")
    println("-"^50)

    println("Running $J simulations...")
    results = simulate_batch(model_func, params, target_trial.rewards, J)
    isempty(results) && (println("No valid simulations generated"); return nothing)

    # Extract data from results
    c1s = [r.choice1 for r in results if !r.timeout]
    rt1s = [r.rt1 for r in results if !r.timeout]
    c2s = [r.choice2 for r in results if !r.timeout]
    rt2s = [r.rt2 for r in results if !r.timeout]
    
    isempty(c1s) && (println("No valid simulations generated"); return nothing)

    println("Generated $(length(c1s)) valid samples")
    # Immediate termination: second-stage decision in one step (rt2 <= T2 + threshold)
    config = get_model_config(MODEL_NAME)
    T2_idx = findfirst(==("T2"), config.param_names)
    T2 = T2_idx !== nothing ? params[T2_idx] : 0.0
    n_immediate = T2_idx !== nothing ? count(rt2s .<= T2 + IMMEDIATE_MS_THRESHOLD) : 0
    pct_immediate = length(rt2s) > 0 ? round(100 * n_immediate / length(rt2s), digits=1) : 0.0
    println("Immediate termination (rt2 <= T2 + $(IMMEDIATE_MS_THRESHOLD) ms): $n_immediate / $(length(rt2s)) ($pct_immediate%)")
    choice1_dist = Dict(c => count(==(c), c1s) for c in unique(c1s))
    choice2_dist = Dict(c => count(==(c), c2s) for c in unique(c2s))
    println("Overall Choice1 distribution: $choice1_dist")
    println("Overall Choice2 distribution: $choice2_dist")

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
        :param_name => param_name, :params => params,
        :all_rt1s => rt1s, :all_rt2s => rt2s,
        :matching_rt1s => mrt1, :matching_rt2s => mrt2,
        :rt1_percentile => p1, :rt2_percentile => p2,
        :n_total => length(c1s), :n_matching => length(idx),
        :n_immediate_termination => n_immediate,
        :pct_immediate_termination => pct_immediate,
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
            # if Z[j, i] < 1e-8
            #     Z[j, i] = 0.0
            # elseif Z[j, i] > 1e-8
            #     Z[j, i] = 1
            # end
        end
    end
    return Z
end

function _create_colorbar_plot(cb_ticks, cb_labels, zmin, zmax)
    n = 20
    dummy_z = reshape(collect(range(zmin, zmax, length=n*n)), n, n)
    dummy_x = collect(range(0, 1, length=n))
    dummy_y = collect(range(0, 1, length=n))
    
    cb_width = 80
    cb_height = 500
    
    cb_plot = contourf(dummy_x, dummy_y, dummy_z;
                       fill=true, c=:viridis, levels=18,
                       colorbar=true,
                       colorbar_ticks=(cb_ticks, cb_labels),
                       framestyle=:box,
                       xticks=false, yticks=false,
                       size=(cb_width, cb_height),
                       margin=5Plots.mm,
                       left_margin=0Plots.mm,
                       right_margin=25Plots.mm, 
                       top_margin=5Plots.mm,
                       bottom_margin=5Plots.mm)
    return cb_plot
end

function _create_legend_plot(legend_items::Vector{Tuple{String, Symbol, Symbol}})
    plt = plot(framestyle=:none, xticks=false, yticks=false, 
               size=(200, 150), margin=0Plots.mm, legend=:outertopright)
    
    for (label, color, style) in legend_items
        if style == :vline || style == :hline
            plot!(plt, [NaN], [NaN]; 
                  label=label, color=color, linestyle=:solid, linewidth=2)
        elseif style == :dash
            plot!(plt, [NaN], [NaN]; 
                  label=label, color=color, linestyle=:dash, linewidth=2)
        elseif style == :rect
            scatter!(plt, [NaN], [NaN]; 
                    label=label, color=color, markershape=:rect, 
                    markerstrokewidth=0, ms=8)
        else
            plot!(plt, [NaN], [NaN]; 
                  label=label, color=color, linestyle=:solid, linewidth=2)
        end
    end
    
    return plt
end

function _plot_tree(trial::Trial)
    root_x, root_y = 0.5, 0.8
    left_x, left_y = 0.25, 0.4
    right_x, right_y = 0.75, 0.4
    leaf = ((0.125,0.1),(0.375,0.1),(0.625,0.1),(0.875,0.1))

    root_size = 10
    branch_size = 10
    leaf_size = 10

    plt = plot(legend=false, framestyle=:none, xticks=false, yticks=false, 
               size=(300,200), 
               xlim=(-0.15, 1.0), ylim=(0.0, 1.0),
               margin=5Plots.mm,
               left_margin=15Plots.mm,
               right_margin=5Plots.mm,
               top_margin=5Plots.mm,
               bottom_margin=5Plots.mm)
    
    scatter!(plt, [root_x], [root_y]; markershape=:circle, ms=root_size, color=:black, markerstrokewidth=0)
    scatter!(plt, [left_x, right_x], [left_y, right_y]; markershape=:circle, ms=branch_size, color=:black, markerstrokewidth=0)
    scatter!(plt, [l[1] for l in leaf], [l[2] for l in leaf]; markershape=:circle, ms=leaf_size, color=:black, markerstrokewidth=0)

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
            scatter!(plt, [leaf[1][1]], [leaf[1][2]]; markershape=:circle, ms=leaf_size, color=:red, markerstrokewidth=0)
        else
            plot!(plt, [left_x, leaf[2][1]], [left_y, leaf[2][2]]; color=:red, linewidth=3, label="")
            scatter!(plt, [leaf[2][1]], [leaf[2][2]]; markershape=:circle, ms=leaf_size, color=:red, markerstrokewidth=0)
        end
    else
        plot!(plt, [root_x, right_x], [root_y, right_y]; color=:red, linewidth=3, label="")
        if trial.choice2 == 1
            plot!(plt, [right_x, leaf[3][1]], [right_y, leaf[3][2]]; color=:red, linewidth=3, label="")
            scatter!(plt, [leaf[3][1]], [leaf[3][2]]; markershape=:circle, ms=leaf_size, color=:red, markerstrokewidth=0)
        else
            plot!(plt, [right_x, leaf[4][1]], [right_y, leaf[4][2]]; color=:red, linewidth=3, label="")
            scatter!(plt, [leaf[4][1]], [leaf[4][2]]; markershape=:circle, ms=leaf_size, color=:red, markerstrokewidth=0)
        end
    end
    # annotate!(tree_plot, x, y, text("$(label)", 8, "Arial"))
    plt
end

function _create_rt1_marginal_plot(mrt1s, xr, trial::Trial, Z, xgrid, ygrid)
    if PLOT_LOG_SPACE
        safe_mrt1s = max.(1e-10, mrt1s)
        plot_rt1s = log.(safe_mrt1s)
        plot_trial_rt1 = log(max(1e-10, trial.rt1))
    else
        plot_rt1s = mrt1s
        plot_trial_rt1 = trial.rt1
    end
    if PLOT_LOG_SPACE
        xr_safe = (max(1e-10, xr[1]), max(1e-10, xr[2]))
        plot_xr = (log(xr_safe[1]), log(xr_safe[2]))
        if !isfinite(plot_xr[1]) || !isfinite(plot_xr[2])
            plot_xr = (log(100.0), log(10000.0))
        end
    else
        plot_xr = xr
    end

    rt1_marginal = vec(sum(Z, dims=1))
    dx = xgrid[2] - xgrid[1]
    rt1_marginal ./= sum(rt1_marginal) * dx

    p_top = histogram(
        plot_rt1s;
        bins = 50,
        normalize = :pdf,
        fillalpha = 0.3,
        fillcolor = :lightblue,
        linecolor = :white,
        xlim = plot_xr,
        xlabel = "",
        ylabel = "",
        xticks = false,
        yticks = false,
        showaxis = false,
        legend = false,
        framestyle = :none,
        bottom_margin = 0Plots.mm,
        top_margin = 0Plots.mm,
        left_margin = 0Plots.mm,
        right_margin = 0Plots.mm
    )

    plot!(p_top, xgrid, rt1_marginal;
          color = :darkblue,
          linewidth = 2,
          label = "")

    vline!(p_top, [plot_trial_rt1];
           color = :red,
           linewidth = 2.5,
           linestyle = :solid,
           label = "")

    vline!(p_top, [mean(plot_rt1s)];
           color = :blue,
           linewidth = 1.5,
           linestyle = :dash,
           alpha = 0.7,
           label = "")

    return p_top
end

function _create_rt2_marginal_plot(mrt2s, yr, trial::Trial, Z, xgrid, ygrid)
    if PLOT_LOG_SPACE
        safe_mrt2s = max.(1e-10, mrt2s)
        plot_rt2s = log.(safe_mrt2s)
        plot_trial_rt2 = log(max(1e-10, trial.rt2))
    else
        plot_rt2s = mrt2s
        plot_trial_rt2 = trial.rt2
    end
    if PLOT_LOG_SPACE
        yr_safe = (max(1e-10, yr[1]), max(1e-10, yr[2]))
        plot_yr = (log(yr_safe[1]), log(yr_safe[2]))
        if !isfinite(plot_yr[1]) || !isfinite(plot_yr[2])
            plot_yr = (log(50.0), log(10000.0))
        end
    else
        plot_yr = yr
    end

    rt2_marginal = vec(sum(Z, dims=2))
    dy = ygrid[2] - ygrid[1] 
    rt2_marginal ./= sum(rt2_marginal) * dy

    maxden = maximum(rt2_marginal)
    xlim_marginal = (0, maxden * 1.2)

    p_right = histogram(
        plot_rt2s;
        bins = 50,
        normalize = :pdf,
        orientation = :h,
        xlim = xlim_marginal,
        ylim = plot_yr,
        xlabel = "",
        ylabel = "",
        xticks = false,
        yticks = false,
        showaxis = false,
        linecolor = :white,
        fillalpha = 0.3,
        fillcolor = :lightgreen,
        legend = false,
        framestyle = :none,
        left_margin = 0Plots.mm,
        right_margin = 0Plots.mm,
        top_margin = 0Plots.mm,
        bottom_margin = 0Plots.mm
    )

    plot!(p_right, rt2_marginal, ygrid;
          color = :darkgreen,
          linewidth = 2)

    # Participant RT2 horizontal line
    hline!(p_right, [plot_trial_rt2];
           color = :red,
           linewidth = 2.5,
           linestyle = :solid)

    # Mean RT2 horizontal line
    hline!(p_right, [mean(plot_rt2s)];
           color = :blue,
           linewidth = 1.5,
           linestyle = :dash,
           alpha = 0.7)

    return p_right
end

# ----------------------------- Joint 2D plotting ----------------------------
function create_joint_kde_plots(results::Vector, target_trial::Trial)
    println("\n" * "="^50)
    println("Creating JOINT 2D KDE plots (with marginal distributions)...")
    println("="^50)

    try
        for res in results
            (res === nothing || res[:n_matching] < 10) && continue

            name  = res[:param_name]
            mrt1s = res[:matching_rt1s]
            mrt2s = res[:matching_rt2s]

            nx, ny = 120, 120

            if PLOT_LOG_SPACE
                safe_mrt1s = max.(1e-10, mrt1s)
                safe_mrt2s = max.(1e-10, mrt2s)
                plot_mrt1s = log.(safe_mrt1s)
                plot_mrt2s = log.(safe_mrt2s)
                
                if MANUAL_LOG_RT1_RANGE !== nothing
                    plot_xr = MANUAL_LOG_RT1_RANGE
                    xr = (exp(plot_xr[1]), exp(plot_xr[2]))
                elseif MANUAL_RT1_RANGE !== nothing
                    xr = MANUAL_RT1_RANGE
                    xr_safe = (max(1e-10, xr[1]), max(1e-10, xr[2]))
                    plot_xr = (log(xr_safe[1]), log(xr_safe[2]))
                    if !isfinite(plot_xr[1]) || !isfinite(plot_xr[2])
                        @warn "Non-finite xr values: $xr -> $plot_xr, using safe defaults"
                        plot_xr = (log(100.0), log(10000.0))
                        xr = (100.0, 10000.0)
                    end
                else
                    xr_min = min(minimum(mrt1s), target_trial.rt1)
                    xr_max = max(maximum(mrt1s), target_trial.rt1)
                    xr_padding = (xr_max - xr_min) * 0.1  # 10% padding
                    xr = (max(1.0, xr_min - xr_padding), min(10000.0, xr_max + xr_padding))
                    xr_safe = (max(1e-10, xr[1]), max(1e-10, xr[2]))
                    plot_xr = (log(xr_safe[1]), log(xr_safe[2]))
                end
                
                if MANUAL_LOG_RT2_RANGE !== nothing
                    plot_yr = MANUAL_LOG_RT2_RANGE
                    yr = (exp(plot_yr[1]), exp(plot_yr[2]))
                elseif MANUAL_RT2_RANGE !== nothing
                    yr = MANUAL_RT2_RANGE
                    yr_safe = (max(1e-10, yr[1]), max(1e-10, yr[2]))
                    plot_yr = (log(yr_safe[1]), log(yr_safe[2]))
                    if !isfinite(plot_yr[1]) || !isfinite(plot_yr[2])
                        @warn "Non-finite yr values: $yr -> $plot_yr, using safe defaults"
                        plot_yr = (log(50.0), log(10000.0))
                        yr = (50.0, 10000.0)
                    end
                else
                    yr_min = min(minimum(mrt2s), target_trial.rt2)
                    yr_max = max(maximum(mrt2s), target_trial.rt2)
                    yr_padding = (yr_max - yr_min) * 0.1  # 10% padding
                    yr = (max(1.0, yr_min - yr_padding), min(10000.0, yr_max + yr_padding))
                    yr_safe = (max(1e-10, yr[1]), max(1e-10, yr[2]))
                    plot_yr = (log(yr_safe[1]), log(yr_safe[2]))
                end
                
                xgrid = collect(range(plot_xr[1], plot_xr[2], length=nx))
                ygrid = collect(range(plot_yr[1], plot_yr[2], length=ny))
                Z = compute_kde2d_grid(plot_mrt1s, plot_mrt2s, xgrid, ygrid;
                                       kde_mode=KDE_MODE, bw_rule=BW_RULE)
            else
                plot_mrt1s = mrt1s
                plot_mrt2s = mrt2s
                
                if MANUAL_RT1_RANGE !== nothing
                    xr = MANUAL_RT1_RANGE
                else
                    xr_min = min(minimum(mrt1s), target_trial.rt1)
                    xr_max = max(maximum(mrt1s), target_trial.rt1)
                    xr_padding = (xr_max - xr_min) * 0.1  # 10% padding
                    xr = (max(100.0, xr_min - xr_padding), min(10000.0, xr_max + xr_padding))
                end
                
                if MANUAL_RT2_RANGE !== nothing
                    yr = MANUAL_RT2_RANGE
                else
                    yr_min = min(minimum(mrt2s), target_trial.rt2)
                    yr_max = max(maximum(mrt2s), target_trial.rt2)
                    yr_padding = (yr_max - yr_min) * 0.1  # 10% padding
                    yr = (max(50.0, yr_min - yr_padding), min(10000.0, yr_max + yr_padding))
                end
                
                plot_xr = xr
                plot_yr = yr
                xgrid = collect(range(xr[1], xr[2], length=nx))
                ygrid = collect(range(yr[1], yr[2], length=ny))
                Z = compute_kde2d_grid(plot_mrt1s, plot_mrt2s, xgrid, ygrid;
                                       kde_mode=KDE_MODE, bw_rule=BW_RULE)
            end

            xlabel_text = PLOT_LOG_SPACE ? "log(RT1)" : "First stage RT (ms)"
            ylabel_text = PLOT_LOG_SPACE ? "log(RT2)" : "Second stage RT (ms)"

            p_top   = _create_rt1_marginal_plot(mrt1s, xr, target_trial, Z, xgrid, ygrid)
            p_right = _create_rt2_marginal_plot(mrt2s, yr, target_trial, Z, xgrid, ygrid)

            p_main = heatmap(
                xgrid, ygrid, Z;
                c        = :viridis,
                colorbar = false,
                xlabel   = xlabel_text,
                ylabel   = ylabel_text,
                xlim     = plot_xr,
                ylim     = plot_yr,
                title    = "",
                legend   = false,
                guidefontsize = 9,
                top_margin    = 0Plots.mm,
                right_margin  = 0Plots.mm,
                left_margin   = 2Plots.mm,   
                bottom_margin = 2Plots.mm   
            )

            if PLOT_LOG_SPACE
                safe_mrt1s = max.(1e-10, mrt1s)
                safe_mrt2s = max.(1e-10, mrt2s)
                safe_rt1 = max(1e-10, target_trial.rt1)
                safe_rt2 = max(1e-10, target_trial.rt2)
                scatter!(p_main, log.(safe_mrt1s), log.(safe_mrt2s);
                         markershape=:x, ms=1.5, ma=0.2, color=:white, label="Simulated samples")
                # Highlight human reaction time data
                scatter!(p_main, [log(safe_rt1)], [log(safe_rt2)];
                         markershape=:star5, ms=12, color=:red, markerstrokewidth=2, 
                         markerstrokecolor=:white, label="Human RT")
            else
                scatter!(p_main, mrt1s, mrt2s;
                         markershape=:x, ms=1.5, ma=0.2, color=:white, label="Simulated samples")
                # Highlight human reaction time data
                scatter!(p_main, [target_trial.rt1], [target_trial.rt2];
                         markershape=:star5, ms=12, color=:red, markerstrokewidth=2,
                         markerstrokecolor=:white, label="Human RT")
            end

            p_blank = plot(framestyle=:none, xticks=false, yticks=false,
                          legend=false, margin=0Plots.mm)

            layout = Plots.grid(2, 2;
                                heights = [0.2, 0.8],
                                widths  = [0.75, 0.25], 
                                hgap    = 1.0,
                                vgap    = 1.0)

            fig = plot(
                p_top,   p_blank, 
                p_main,  p_right;
                layout = layout,
                size   = (450, 400),
                dpi    = 500,
                margin = 0Plots.mm
            )

            figures_dir = joinpath(@__DIR__, "figures")
            if !isdir(figures_dir)
                mkpath(figures_dir)
            end

            base_id = USE_MANUAL_REWARDS ? "synthetic_rewards" : PARTICIPANT_ID
            base_trial = USE_MANUAL_REWARDS ? "trial_auto" : string(TRIAL_INDEX)
            fname = "participant_$(base_id)_trial_$(base_trial)_joint2d_$(replace(name, ' '=>'_')).png"
            full_path = joinpath(figures_dir, fname)
            savefig(fig, full_path)
            println(" Saved joint plot: $full_path")
        end
    catch e
        println("Joint plotting failed: $e")
        rethrow(e)
    end
end

# ----------------------------- Main -----------------------------------------
try
    config = get_model_config(MODEL_NAME)
    model_func = config.model_function

    target_trial = Trial(
        "init",
        Float64[],   # rewards
        0,           # choice1
        0,           # choice2
        0.0,         # rt1
        0.0,         # rt2
        Float64[]    # path
    )
    all_results = Any[]

    if USE_MANUAL_REWARDS
        println("\nUsing MANUAL_REWARDS mode (synthetic trial)...")

        @assert length(MANUAL_REWARDS) == 6 "MANUAL_REWARDS must have length 6: [R_L, R_R, R_LL, R_LR, R_RL, R_RR]"
        @assert length(MANUAL_PARAMS) == length(config.param_names) "MANUAL_PARAMS length must match model parameter count"

        # Simulate one trial to define the target choice and RTs
        result = model_func(MANUAL_PARAMS, MANUAL_REWARDS)
        if result.timeout
            error("Simulation with MANUAL_REWARDS and MANUAL_PARAMS timed out; cannot build target trial.")
        end

        path_values = MANUAL_REWARDS[3:6]  # final leaf rewards
        target_trial = Trial(
            "synthetic_manual",
            MANUAL_REWARDS,
            result.choice1,
            result.choice2,
            result.rt1,
            result.rt2,
            path_values
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
        try
            pda_param_dict = load_fitted_parameters(PDA_RESULTS_FILE, MODEL_NAME)
            !haskey(pda_param_dict, PARTICIPANT_ID) && error("Participant $PARTICIPANT_ID not found in PDA results")
            pda_params = pda_param_dict[PARTICIPANT_ID]
            println("PDA parameters loaded: $pda_params")
            
            # Calculate and print log likelihood of participant joint reaction time
            log_likelihood, used_eps = pda_loglike_single_trial(pda_params, target_trial, model_func;
                                                       J=J, kde_mode=KDE_MODE, bw_rule=BW_RULE,
                                                       logRT=LOG_RT, eps_floor=EPS_FLOOR)
            println("\nLog likelihood of participant joint reaction time: $log_likelihood")
            println("   Used eps_floor: $used_eps")
            
            push!(all_results, analyze_parameter_set("PDA Fitted", pda_params, target_trial, model_func))
        catch e
            println("Could not load PDA parameters: $e"); push!(all_results, nothing)
        end
    end

    create_joint_kde_plots(all_results, target_trial)

    println("\n" * "="^80)
    if USE_MANUAL_REWARDS
        println("Analysis Summary for Synthetic Manual Trial")
    else
        println("Analysis Summary for Participant $PARTICIPANT_ID, Trial $TRIAL_INDEX")
    end
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
    println("2. If using participant mode, ensure results files exist and have correct participant IDs")
    println("3. Check that PARTICIPANT_ID and TRIAL_INDEX are valid (participant mode)")
    println("4. If USE_MANUAL_REWARDS = true, verify MANUAL_REWARDS and MANUAL_PARAMS shapes")
    println("5. Verify model parameter structure matches MANUAL_PARAMS or CSV columns")
end