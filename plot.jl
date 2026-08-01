using DataFrames
using CSV
using Plots
using Dates
using Statistics
using StatsBase

using Statistics
using Printf
using Plots
using DataFrames

# --- helper: MAE/RMSE ---
rmse(x) = sqrt(mean(x .^ 2))

"""
Compute total log-likelihood for a set of trials using either :analytic or :pda,
while sweeping one parameter (param_idx) over grid values.
To keep it practical, you can subsample first N trials.
"""
function likelihood_slice(trials, base_params, model_function;
                          param_idx::Int=1,
                          grid::AbstractVector{<:Real},
                          method::Symbol=:analytic,
                          Ntrials::Int=50,
                          # PDA options
                          J::Int=1000, min_sims::Int=1000, min_matching::Int=100,
                          max_sims::Int=100000, eps_floor::Float64=1e-8,
                          bw_scale::Float64=1.0, logRT::Bool=true)

    n_use = min(Ntrials, length(trials))
    trials_use = trials[1:n_use]

    ll = similar(collect(grid), Float64)
    tmp_params = copy(base_params)

    for (k, val) in enumerate(grid)
        tmp_params[param_idx] = Float64(val)
        s = 0.0
        for tr in trials_use
            if method == :analytic
                s += loglik_trial_stagewise(tr, tmp_params)
            elseif method == :pda
                l, _, _ = loglik_trial_pda(tr, tmp_params, model_function;
                                           J=J, min_sims=min_sims, min_matching=min_matching,
                                           max_sims=max_sims, eps_floor=eps_floor,
                                           bw_scale=bw_scale, logRT=logRT)
                s += l
            else
                error("Unknown method: $method")
            end
        end
        ll[k] = s
    end
    return ll
end


"""
Fig2: 2×2 panels to show multi-faceted agreement between PDA and Analytical likelihoods.

Panels:
A) PDA vs Analytical (hexbin/scatter) + y=x + MAE/RMSE
B) Bland–Altman: (mean ll) vs (diff) + bias + LoA
C) Likelihood slice: sweep one parameter, compare total LL curves
D) Error across difficulty (diff2 if exists else rt2 quantiles)
"""
function plot_fig2_panels(df::DataFrame, wid::String;
                          trials::Union{Nothing,Vector}=nothing,
                          params::Union{Nothing,Vector{Float64}}=nothing,
                          model_function::Union{Nothing,Function}=nothing,
                          # slice settings
                          slice_param_idx::Int=1,
                          slice_param_name::String="d1",
                          slice_span_frac::Float64=0.25,
                          slice_grid_n::Int=21,
                          slice_Ntrials::Int=50,
                          # PDA settings for slice
                          J::Int=1000, min_sims::Int=1000, min_matching::Int=100,
                          max_sims::Int=100000, eps_floor::Float64=1e-8,
                          bw_scale::Float64=1.0, logRT::Bool=true,
                          output_file::Union{Nothing,String}=nothing)

    # --------- filter (optional) -------------
    # keep only finite likelihoods
    mask = isfinite.(df.pda_loglik) .& isfinite.(df.analytical_loglik)
    dff = df[mask, :]

    pda_lik = dff.pda_loglik
    ana_lik = dff.analytical_loglik
    diff = dff.loglik_diff
    mean_ll = dff.mean_ll
    abs_err = dff.abs_err

    mae = mean(abs_err)
    r = cor(pda_lik, ana_lik)

    # common lims for panel A
    allv = vcat(pda_lik, ana_lik)
    vmin, vmax = minimum(allv), maximum(allv)
    mrg = 0.05 * (vmax - vmin + eps())
    lims = (vmin - mrg, vmax + mrg)

    # ---------------- Panel A: PDA vs Analytical ----------------
    # Use hexbin if many points; fall back to scatter if few
    pA = plot(fontfamily="Arial", size=(350, 320), dpi=300)
    if nrow(dff) > 800
        hexbin!(pA, pda_lik, ana_lik;
                xlims=lims, ylims=lims,
                xlabel="PDA log-likelihood",
                ylabel="Analytical log-likelihood",
                legend=false)
    else
        scatter!(pA, pda_lik, ana_lik;
                 ms=3, alpha=0.5, color=:blue,
                 xlims=lims, ylims=lims,
                 xlabel="PDA log-likelihood",
                 ylabel="Analytical log-likelihood",
                 legend=false)
    end
    plot!(pA, [lims[1], lims[2]], [lims[1], lims[2]];
          lw=1.5, ls=:dot, color=:gray)

    annA = @sprintf("r = %.3f\nMAE = %.3g\nRMSE = %.3g", r, mae, rmse(diff))
    annotate!(pA, lims[1] + 0.06*(lims[2]-lims[1]), lims[2] - 0.12*(lims[2]-lims[1]),
              text(annA, 9, :left, "Arial"))

    # ---------------- Panel B: Bland–Altman ----------------
    bias = mean(diff)
    sd = std(diff)
    loa_hi = bias + 1.96*sd
    loa_lo = bias - 1.96*sd

    pB = scatter(mean_ll, diff;
                 ms=3, alpha=0.5, color=:purple,
                 xlabel="Mean log-likelihood",
                 ylabel="Difference (PDA − Analytical)",
                 legend=false,
                 fontfamily="Arial", size=(350, 320), dpi=300)

    hline!(pB, [bias]; lw=2, color=:black, ls=:solid)
    hline!(pB, [loa_hi, loa_lo]; lw=1.5, color=:gray, ls=:dash)

    annB = @sprintf("bias = %.3g\nLoA = [%.3g, %.3g]", bias, loa_lo, loa_hi)
    xB = minimum(mean_ll) + 0.06*(maximum(mean_ll)-minimum(mean_ll)+eps())
    yB = maximum(diff) - 0.12*(maximum(diff)-minimum(diff)+eps())
    annotate!(pB, xB, yB, text(annB, 9, :left, "Arial"))

    # ---------------- Panel C: Likelihood slice ----------------
    # Only if trials/params/model_function are provided
    pC = plot(fontfamily="Arial", size=(350, 320), dpi=300,
              xlabel=slice_param_name, ylabel="Total log-likelihood",
              legend=:bottomright)

    if trials !== nothing && params !== nothing && model_function !== nothing
        θ0 = params[slice_param_idx]
        span = slice_span_frac * (abs(θ0) + 1e-6)
        grid = range(θ0 - span, θ0 + span, length=slice_grid_n)

        ll_ana = likelihood_slice(trials, params, model_function;
                                  param_idx=slice_param_idx, grid=grid,
                                  method=:analytic, Ntrials=slice_Ntrials)

        ll_pda = likelihood_slice(trials, params, model_function;
                                  param_idx=slice_param_idx, grid=grid,
                                  method=:pda, Ntrials=slice_Ntrials,
                                  J=J, min_sims=min_sims, min_matching=min_matching,
                                  max_sims=max_sims, eps_floor=eps_floor,
                                  bw_scale=bw_scale, logRT=logRT)

        plot!(pC, grid, ll_ana; lw=2, label="Analytical")
        plot!(pC, grid, ll_pda; lw=2, ls=:dash, label="PDA")

        vline!(pC, [θ0]; lw=1.5, ls=:dot, color=:gray, label=false)

        annC = @sprintf("slice over %s\nNtrials = %d", slice_param_name, min(slice_Ntrials, length(trials)))
        annotate!(pC, grid[1] + 0.05*(grid[end]-grid[1]),
                  maximum(vcat(ll_ana, ll_pda)) - 0.10*(maximum(vcat(ll_ana,ll_pda))-minimum(vcat(ll_ana,ll_pda))+eps()),
                  text(annC, 9, :left, "Arial"))
    else
        plot!(pC, [0,1], [0,1], label=false)
        annotate!(pC, 0.5, 0.5, text("Slice panel requires\n(trials, params, model_function)", 10, :center, "Arial"))
    end

    # ---------------- Panel D: Error across difficulty (or fallback) ----------------
    pD = plot(fontfamily="Arial", size=(350, 320), dpi=300,
              xlabel="", ylabel="Mean |error| (|PDA − Analytical|)",
              legend=false)

    if :diff2 in propertynames(dff) && any(.!ismissing.(dff.diff2))
        d2 = dff.diff2
        mask2 = .!ismissing.(d2)
        tmp = DataFrame(diff2 = Float64.(d2[mask2]), abs_err = abs_err[mask2])

        # bin diff2 by rounding to avoid too many categories
        tmp.diff2_bin = round.(tmp.diff2; digits=2)

        g = groupby(tmp, :diff2_bin)
        xs = [first(key).diff2_bin for key in keys(g)]
        ys = [mean(gr.abs_err) for gr in g]
        es = [std(gr.abs_err) / sqrt(nrow(gr)) for gr in g]  # SE

        # sort by x
        ord = sortperm(xs)
        xs, ys, es = xs[ord], ys[ord], es[ord]

        scatter!(pD, xs, ys; ms=4, alpha=0.9, color=:darkgreen)
        plot!(pD, xs, ys; lw=2, color=:darkgreen)
        # error bars
        for i in eachindex(xs)
            plot!(pD, [xs[i], xs[i]], [ys[i]-es[i], ys[i]+es[i]]; lw=1, color=:darkgreen, alpha=0.8)
        end
        xlabel!(pD, "Difficulty (diff2)")
    else
        # fallback: RT2 quantiles
        rt2 = dff.rt2
        q = quantile(rt2, [0.0, 0.25, 0.5, 0.75, 1.0])
        bin = similar(rt2, Int)
        for i in eachindex(rt2)
            bin[i] = rt2[i] <= q[2] ? 1 :
                     rt2[i] <= q[3] ? 2 :
                     rt2[i] <= q[4] ? 3 : 4
        end
        tmp = DataFrame(bin = bin, abs_err = abs_err)
        g = groupby(tmp, :bin)
        xs = collect(1:length(g))
        ys = [mean(gr.abs_err) for gr in g]
        es = [std(gr.abs_err)/sqrt(nrow(gr)) for gr in g]

        scatter!(pD, xs, ys; ms=5, alpha=0.9, color=:darkgreen)
        plot!(pD, xs, ys; lw=2, color=:darkgreen)
        for i in eachindex(xs)
            plot!(pD, [xs[i], xs[i]], [ys[i]-es[i], ys[i]+es[i]]; lw=1, color=:darkgreen, alpha=0.8)
        end
        xlabel!(pD, "RT2 quantile bin (fallback)")
    end

    # ---------------- Combine ----------------
    fig = plot(pA, pB, pC, pD;
               layout=(2,2),
               size=(900, 700),
               dpi=300,
               left_margin=4Plots.mm, right_margin=2Plots.mm,
               top_margin=2Plots.mm, bottom_margin=3Plots.mm)

    # add panel labels
    # (rough placement; adjust if needed)
    plot!(fig, plot_title = "")
    annotate!(fig[1], lims[1] + 0.01*(lims[2]-lims[1]), lims[2] - 0.01*(lims[2]-lims[1]), text("A", 12, :left, "Arial"))
    annotate!(fig[2], minimum(mean_ll) + 0.01*(maximum(mean_ll)-minimum(mean_ll)+eps()),
              maximum(diff) - 0.01*(maximum(diff)-minimum(diff)+eps()), text("B", 12, :left, "Arial"))
    annotate!(fig[3], 0.05, 0.95, text("C", 12, :left, "Arial"))  # relative-ish placeholder
    annotate!(fig[4], 0.05, 0.95, text("D", 12, :left, "Arial"))

    if output_file !== nothing
        mkpath(dirname(output_file))
        savefig(fig, output_file)
        println("Fig2 saved to: $output_file")
    end

    display(fig)
    return fig
end

# ----------------- Plot comparison results -----------------
function plot_likelihood_comparison(df::DataFrame, wid::String; 
                                   output_file::Union{String,Nothing} = nothing)
    """
    Plot comparison between PDA and analytical likelihoods.
    
    First subplot: Scatter plot of PDA vs Analytical likelihoods with fitted line
    Second subplot: Difference (PDA - Analytical) vs trial index
    """
    
    # Extract data and filter out trials where analytical_loglik used eps_floor
    # Check if analytical_used_eps column exists (for backward compatibility)
    if hasproperty(df, :analytical_used_eps)
        valid_mask = .!df.analytical_used_eps
        pda_lik = df.pda_loglik[valid_mask]
        analytical_lik = df.analytical_loglik[valid_mask]
        diff = df.loglik_diff[valid_mask]
        trial_idx = df.trial_idx[valid_mask]
        n_excluded = sum(df.analytical_used_eps)
        if n_excluded > 0
            println("    Excluding $n_excluded trials where analytical_loglik used eps_floor from plots")
        end
    else
        # If column doesn't exist, use all data (backward compatibility)
        pda_lik = df.pda_loglik
        analytical_lik = df.analytical_loglik
        diff = df.loglik_diff
        trial_idx = df.trial_idx
    end
    
    # Use filtered data for both plots (exclude trials where analytical_loglik used eps_floor)
    
    # Fit linear regression: analytical = slope * pda + intercept
    # Using least squares: y = X * β where X = [pda_lik ones]
    X = hcat(pda_lik, ones(length(pda_lik)))
    β = X \ analytical_lik
    slope = β[1]
    intercept = β[2]
    
    # Calculate R²
    y_pred = X * β
    ss_res = sum((analytical_lik .- y_pred).^2)
    ss_tot = sum((analytical_lik .- mean(analytical_lik)).^2)
    r_squared = 1 - ss_res / ss_tot
    
    # Determine common axis limits for square plot
    all_values = vcat(pda_lik, analytical_lik)
    min_val = minimum(all_values)
    max_val = maximum(all_values)
    # Add small margin
    margin = (max_val - min_val) * 0.05
    lim_min = min_val - margin
    lim_max = max_val + margin
    
    # Create fitted line points
    line_x = [lim_min, lim_max]
    line_y = slope .* line_x .+ intercept
    
    # Create plots
    # Subplot 1: Scatter plot with fitted line
    p1 = scatter(pda_lik, analytical_lik,
                 label="Trials",
                 alpha=0.6,
                 markersize=5,  # Increased point size
                 color=:blue,
                 xlabel="PDA Log-likelihood",
                 ylabel="Analytical Log-likelihood",
            #      title="PDA vs Analytical Likelihoods",
                 titlefontsize=10,
                 xlims=(lim_min, lim_max),
                 ylims=(lim_min, lim_max),
                 aspect_ratio=:equal,  # Make it square
                 legend=:topleft,
                 bottom_margin=2Plots.mm,  # Reduced margin
                 left_margin=2Plots.mm,    # Reduced margin
                 fontfamily="Arial",
                 guidefont=font(9, "Arial"),      # Larger axis label font
                 tickfont=font(9, "Arial"),        # Larger tick font with Arial
                 legendfont=font(8, "Arial"),       # Smaller legend font
                 legendfontsize=8,
                 size=(200, 200))                   # Square size
    
    # Explicitly set axis label fonts after creation
    plot!(p1, 
          xguidefont=font(9, "Arial"),
          yguidefont=font(9, "Arial"))
    
    # Add fitted line
    plot!(p1, line_x, line_y,
          label="Fitted line",
          linewidth=2,
          color=:red,
          linestyle=:dash)
    
    # Add diagonal reference line (y = x)
    plot!(p1, [lim_min, lim_max], [lim_min, lim_max],
          label="y = x",
          linewidth=1,
          color=:gray,
          linestyle=:dot,
          alpha=0.5)
    
    # Add text annotation with slope and intercept (bottom-right to avoid legend overlap)
    text_str = "y = $(round(slope, digits=3))x + $(round(intercept, digits=3))\nR² = $(round(r_squared, digits=3))"
    annotate!(p1, lim_max - (lim_max - lim_min) * 0.05, lim_min + (lim_max - lim_min) * 0.15,
              text(text_str, font(8, "Arial"), :right, :black))  # Smaller text size (8) with Arial font
    
    # Subplot 2: Difference plot (trial index vs difference)
    # Use filtered data (same as p1) to exclude trials where analytical_loglik used eps_floor
    p2 = scatter(trial_idx, diff,
                 label="Difference",
                 alpha=0.6,
                 markersize=5,
                 color=:purple,
                 xlabel="Trial Index",
                 ylabel="Difference (PDA - Analytical)",
            #      title="Likelihood Difference by Trial",
                 titlefontsize=10,
                 legend=false,
                 bottom_margin=2Plots.mm,  # Reduced margin
                 left_margin=2Plots.mm,   # Reduced margin
                 fontfamily="Arial",
                 guidefont=font(9, "Arial"),      # Larger axis label font
                 tickfont=font(9, "Arial"),        # Larger tick font
                 size=(300, 200))                   # Wider plot for trial index
    
    # Explicitly set axis label fonts after creation
    plot!(p2,
          xguidefont=font(9, "Arial"),
          yguidefont=font(9, "Arial"))
    
    # Add horizontal line at y = 0
    hline!(p2, [0.0],
           linewidth=1,
           color=:gray,
           linestyle=:dash,
           alpha=0.5)
    
    # Combine plots - p1 is square, p2 is wider (trial index plot)
    # Use compact layout with reduced margins and spacing
    p_combined = plot(p1, p2, 
                      layout=(1, 2), 
                      size=(600, 250),  # Compact size
                      dpi=300,
                      fontfamily="Arial",
                      guidefont=font(11, "Arial"),      # Axis labels
                      tickfont=font(9, "Arial"),        # Tick numbers with Arial
                      legendfont=font(8, "Arial"),       # Legend with smaller size
                      left_margin=2Plots.mm,            # Reduced left margin
                      right_margin=2Plots.mm,           # Reduced right margin
                      top_margin=2Plots.mm,             # Reduced top margin
                      bottom_margin=2Plots.mm,          # Reduced bottom margin
                      plot_title="")                     # No title to save space
    
    # Save plot
    if output_file === nothing
        output_file = "pda/figures/$(wid)_likelihood_comparison_$(Dates.format(now(), "yyyymmdd_HHMMSS")).png"
    end
    
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    savefig(p_combined, output_file)
    println("\n[4] Plot saved to: $output_file")
    
    # Display plot
    display(p_combined)
    
    return p_combined
end

# ----------------- Plot CDF/PDF comparison for K-S test -----------------
function plot_cdf_pdf_comparison(wid::String, trial_idx::Int,
                                 t_values::Vector{Float64},
                                 rt1_sim::Vector{Float64}, rt2_sim::Vector{Float64},
                                 pdf_rt1::Vector{Float64}, pdf_rt2::Vector{Float64},
                                 pdf_rt1_analytical::Vector{Float64}, pdf_rt2_analytical::Vector{Float64},
                                 cdf_rt1_analytical::Vector{Float64}, cdf_rt2_analytical::Vector{Float64},
                                 cdf_rt1_empirical::Vector{Float64}, cdf_rt2_empirical::Vector{Float64},
                                 cdf_rt1_pda::Vector{Float64}, cdf_rt2_pda::Vector{Float64},
                                 target_trial::Trial,
                                 ks_rt1, ks_rt2;
                                 log_scale::Bool=false,
                                 output_file::Union{String,Nothing}=nothing)
    """
    Plot 2x2 subplot comparing PDFs and CDFs for RT1 and RT2.
    
    Top row: PDFs (RT1 left, RT2 right)
    Bottom row: CDFs (RT1 left, RT2 right)
    """
    
    # Set scale options based on log_scale parameter
    pdf_yscale = log_scale ? :log10 : :identity
    pdf_xscale = log_scale ? :log10 : :identity
    cdf_yscale = log_scale ? :log10 : :identity
    
    # Create 2x2 subplot layout
    plt = plot(layout=@layout([a b; c d]), size=(1000, 1000), 
               dpi=300, title="$(wid), Trial $trial_idx" * (log_scale ? " (Log Scale)" : ""))
    
    # Top left: RT1 PDF
    histogram!(plt[1], rt1_sim, bins=30, normalize=:pdf, alpha=0.2, color=:lightblue, label="Empirical histogram")
    plot!(plt[1], t_values, pdf_rt1, linewidth=2, linestyle=:dash, color=:purple, alpha=0.8, label="PDA")
    plot!(plt[1], t_values, pdf_rt1_analytical, linewidth=2.5, color=:blue, label="Analytical PDF")
    vline!(plt[1], [target_trial.rt1], linewidth=1.5, linestyle=:dot, color=:green, label="Observed RT1")
    plot!(plt[1], title="RT1 PDF", xlabel="Time (ms)", ylabel=log_scale ? "Density (log10)" : "Density", 
          legend=:topright, fontsize=9, yscale=pdf_yscale, xscale=pdf_xscale)
    
    # Top right: RT2 PDF
    histogram!(plt[2], rt2_sim, bins=30, normalize=:pdf, alpha=0.2, color=:lightcoral, label="Empirical histogram")
    plot!(plt[2], t_values, pdf_rt2, linewidth=2, linestyle=:dash, color=:purple, alpha=0.8, label="PDA")
    plot!(plt[2], t_values, pdf_rt2_analytical, linewidth=2.5, color=:blue, label="Analytical PDF")
    vline!(plt[2], [target_trial.rt2], linewidth=1.5, linestyle=:dot, color=:green, label="Observed RT2")
    plot!(plt[2], title="RT2 PDF", xlabel="Time (ms)", ylabel=log_scale ? "Density (log10)" : "Density", 
          legend=:topright, fontsize=9, yscale=pdf_yscale, xscale=pdf_xscale)
    
    # Bottom left: RT1 CDF
    plot!(plt[3], t_values, cdf_rt1_analytical, label="Analytical CDF", linewidth=2.5, color=:blue)
    plot!(plt[3], t_values, cdf_rt1_pda, label="PDA CDF", 
          linewidth=2, linestyle=:dash, color=:purple, alpha=0.8)
    vline!(plt[3], [target_trial.rt1], label="Observed RT1", linewidth=1.5, linestyle=:dot, color=:green)
    vline!(plt[3], [ks_rt1.max_diff_t], label="Max diff", linewidth=1, linestyle=:dashdot, color=:black, alpha=0.5)
    annotate!(plt[3], 0.05, 0.95, text("D=$(round(ks_rt1.max_diff, digits=4)), p=$(round(ks_rt1.p_value, digits=4))", 
             :left, :top, 8))
    plot!(plt[3], title="RT1 CDF", xlabel="Time (ms)", ylabel=log_scale ? "CDF (log10)" : "CDF", 
          legend=:bottomright, fontsize=9, yscale=cdf_yscale, xscale=pdf_xscale, ylims=(0.0, 1.0))
    
    # Bottom right: RT2 CDF
    plot!(plt[4], t_values, cdf_rt2_analytical, label="Analytical CDF", linewidth=2.5, color=:blue)
    plot!(plt[4], t_values, cdf_rt2_pda, label="PDA CDF", 
          linewidth=2, linestyle=:dash, color=:purple, alpha=0.8)
    vline!(plt[4], [target_trial.rt2], label="Observed RT2", linewidth=1.5, linestyle=:dot, color=:green)
    vline!(plt[4], [ks_rt2.max_diff_t], label="Max diff", linewidth=1, linestyle=:dashdot, color=:black, alpha=0.5)
    annotate!(plt[4], 0.05, 0.95, text("D=$(round(ks_rt2.max_diff, digits=4)), p=$(round(ks_rt2.p_value, digits=4))", 
             :left, :top, 8))
    plot!(plt[4], title="RT2 CDF", xlabel="Time (ms)", ylabel=log_scale ? "CDF (log10)" : "CDF", 
          legend=:bottomright, fontsize=9, yscale=cdf_yscale, xscale=pdf_xscale, ylims=(0.0, 1.0))
    
    # Save plot
    if output_file === nothing
        output_dir = "pda/figures"
        if !isdir(output_dir)
            mkpath(output_dir)
        end
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        scale_suffix = log_scale ? "_log" : ""
        output_file = joinpath(output_dir, "cdf_pdf_comparison_$(wid)_trial$(trial_idx)$(scale_suffix)_$timestamp.png")
    end
    
    output_dir_actual = dirname(output_file)
    if !isdir(output_dir_actual) && output_dir_actual != ""
        mkpath(output_dir_actual)
    end
    
    savefig(plt, output_file)
    println("    Saved: $(basename(output_file))")
    
    return plt
end