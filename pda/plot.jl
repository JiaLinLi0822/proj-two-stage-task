using DataFrames
using CSV
using Plots
using Dates

# ----------------- Plot comparison results -----------------
function plot_likelihood_comparison(df::DataFrame, wid::String; 
                                   output_file::Union{String,Nothing} = nothing)
    """
    Plot comparison between PDA and analytical likelihoods.
    
    First subplot: Scatter plot of PDA vs Analytical likelihoods with fitted line
    Second subplot: Difference (PDA - Analytical) vs trial index
    """
    
    # Extract data
    pda_lik = df.pda_loglik
    analytical_lik = df.analytical_loglik
    diff = df.loglik_diff
    trial_idx = df.trial_idx
    
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
                 markersize=3,
                 color=:blue,
                 xlabel="PDA Log-likelihood",
                 ylabel="Analytical Log-likelihood",
                 title="PDA vs Analytical Likelihoods",
                 titlefontsize=10,
                 xlims=(lim_min, lim_max),
                 ylims=(lim_min, lim_max),
                 aspect_ratio=:equal,  # Make it square
                 legend=:topleft,
                 bottom_margin=5Plots.mm,  # Add bottom margin for x-axis labels
                 left_margin=5Plots.mm)    # Add left margin for y-axis labels
    
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
              text(text_str, :right, 10, :black, :right))
    
    # Subplot 2: Difference plot
    p2 = scatter(trial_idx, diff,
                 label="Difference",
                 alpha=0.6,
                 markersize=3,
                 color=:purple,
                 xlabel="Trial Index",
                 ylabel="Difference (PDA - Analytical)",
                 title="Likelihood Difference by Trial",
                 titlefontsize=10,
                 legend=false,
                 bottom_margin=5Plots.mm,
                 left_margin=5Plots.mm)
    
    # Add horizontal line at y = 0
    hline!(p2, [0.0],
           linewidth=1,
           color=:gray,
           linestyle=:dash,
           alpha=0.5)
    
    # Combine plots
    p_combined = plot(p1, p2, layout=(1, 2), size=(800, 350), dpi=300)
    
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