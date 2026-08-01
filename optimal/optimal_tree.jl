using Distributions, LinearAlgebra, StaticArrays, ImageFiltering, ProgressMeter
using ImageFiltering: Pad
using Base.Threads
using Plots

# --- Node and variance-transition definitions ---
# (These are correct and remain unchanged)
struct Node
    mean_r::Float64
    var_r::Float64
    var_x::Float64
end

"""Posterior variance of reward at time t."""
function post_var(p::Node, t::Float64)
    return (p.var_r * p.var_x) / (p.var_x + t * p.var_r)
end

"""Variance of the update (predictive variance) over [t, t+dt]."""
function var_trans(p::Node, t::Float64, dt::Float64)
    vp_t  = post_var(p, t)
    vp_t1 = post_var(p, t + dt)
    return vp_t - vp_t1
end

# --- DP parameter struct (CORRECTED) ---
struct Stage1Params
    zs::Int                # grid size per dimension
    ts::Int                # number of time steps
    z_max::Float64         # range limit for z
    dt::Float64            # time increment
    rho_L::Float64         # correlation in left subtree
    rho_R::Float64         # correlation in right subtree
    cost::Float64          # observation cost per unit time
    t_null::Float64        # decision delay cost
    tol::Float64           # tolerance for tie-breaking
    priors::Node           # (CORRECTED) Priors for the information paths
end

# --- Clark's expected max of two Gaussians ---
const normal = Normal(0, 1)
function clark(mu1, mu2, σ1, σ2, ρ)
    # Added a check for numerical stability
    θ_sq = σ1^2 + σ2^2 - 2*ρ*σ1*σ2
    if θ_sq < 1e-12
        return max(mu1, mu2)
    end
    θ = sqrt(θ_sq)
    δ = (mu1 - mu2) / θ
    return mu1 * cdf(normal, δ) + mu2 * cdf(normal, -δ) + θ * pdf(normal, δ)
end

# --- DP solver for two-stage, four-path tree (CORRECTED) ---
function solve_stage1(p::Stage1Params)
    zs, ts = p.zs, p.ts
    z_grid = range(-p.z_max, p.z_max, length=zs)
    tol = p.tol

    # Precompute 2D mesh for kernels
    X = reshape(repeat(collect(z_grid), inner=zs), zs, zs)
    Y = reshape(repeat(collect(z_grid), outer=zs), zs, zs)

    # Allocate value & decision arrays
    V = Array{Float64}(undef, zs, zs, zs, zs, ts)
    D = Array{UInt8}(undef, zs, zs, zs, zs, ts)

    # --- Terminal boundary at t = ts (CORRECTED) ---
    # Calculate posterior variance at the final time step
    terminal_time = (ts - 1) * p.dt
    terminal_var = post_var(p.priors, terminal_time)
    terminal_sd = sqrt(terminal_var)

    for i in 1:zs, j in 1:zs, k in 1:zs, l in 1:zs
        # (CORRECTED) Use the correct standard deviation for the posterior belief
        lv = clark(z_grid[i], z_grid[j], terminal_sd, terminal_sd, p.rho_L)
        rv = clark(z_grid[k], z_grid[l], terminal_sd, terminal_sd, p.rho_R)
        if lv >= rv
            V[i,j,k,l,ts] = lv - p.t_null; D[i,j,k,l,ts] = 2
        else
            V[i,j,k,l,ts] = rv - p.t_null; D[i,j,k,l,ts] = 3
        end
    end

    # Backward induction
    @showprogress "DP Progress: " for t in (ts-1):-1:1
        time = (t - 1) * p.dt

        # (CORRECTED) Predictive variance for the update kernel (information gain)
        # This fixes the MethodError by calling var_trans with the Node object
        s = var_trans(p.priors, time, p.dt)
        s_LL, s_LR, s_RL, s_RR = s, s, s, s # Assuming identical paths

        # Build 2×2 covariance blocks
        Σ_L = @SMatrix [s_LL  p.rho_L*sqrt(s_LL*s_LR);
                        p.rho_L*sqrt(s_LL*s_LR)  s_LR]
        Σ_R = @SMatrix [s_RL  p.rho_R*sqrt(s_RL*s_RR);
                        p.rho_R*sqrt(s_RL*s_RR)  s_RR]
        invL, detL = inv(Σ_L), det(Σ_L)
        invR, detR = inv(Σ_R), det(Σ_R)

        # Gaussian kernels
        K_L = @. exp(-0.5*(invL[1,1]*X^2 + 2*invL[1,2]*X.*Y + invL[2,2]*Y^2)) / (2*pi*sqrt(detL))
        K_L ./= sum(K_L)
        K_R = @. exp(-0.5*(invR[1,1]*X^2 + 2*invR[1,2]*X.*Y + invR[2,2]*Y^2)) / (2*pi*sqrt(detR))
        K_R ./= sum(K_R)

        # Convolve: first over (LL,LR), then (RL,RR)
        tmp = similar(V[:,:,:,:,t])
        for k in 1:zs, l in 1:zs
            tmp[:,:,k,l] = imfilter(view(V, :, :, k, l, t+1), K_L, Pad(:replicate))
        end
        future = similar(tmp)
        for i in 1:zs, j in 1:zs
            future[i,j,:,:] = imfilter(view(tmp, i, j, :, :), K_R, Pad(:replicate))
        end

        # --- Q-value update and policy (CORRECTED) ---
        # (CORRECTED) Calculate the current posterior standard deviation for Clark's formula
        current_var = post_var(p.priors, time)
        current_sd = sqrt(current_var)

        @inbounds for i in 1:zs, j in 1:zs, k in 1:zs, l in 1:zs
            Qw = future[i,j,k,l] - p.cost*p.dt
            # (CORRECTED) Use the correct posterior standard deviation
            Ql = clark(z_grid[i], z_grid[j], current_sd, current_sd, p.rho_L) - p.t_null
            Qr = clark(z_grid[k], z_grid[l], current_sd, current_sd, p.rho_R) - p.t_null
            if Qw > Ql && Qw > Qr
                V[i,j,k,l,t] = Qw
                D[i,j,k,l,t] = 1
            else
                Δ = Ql - Qr
                if abs(Δ) <= tol
                    # --- tie: choose by grid magnitude ---
                    magL = hypot(z_grid[i], z_grid[j])
                    magR = hypot(z_grid[k], z_grid[l])
                    if magL >= magR
                        V[i,j,k,l,t] = Ql
                        D[i,j,k,l,t] = 2
                    else
                        V[i,j,k,l,t] = Qr
                        D[i,j,k,l,t] = 3
                    end
                elseif Δ > 0
                    V[i,j,k,l,t] = Ql
                    D[i,j,k,l,t] = 2
                else
                    V[i,j,k,l,t] = Qr
                    D[i,j,k,l,t] = 3
                end
            end
        end
    end

    return V, D, z_grid
end

"""
Animate a 2D slice of the value function V over time.

Arguments:
- V: 5D Array (LL, LR, RL, RR, t)
- grid: 1D range of z values
Keyword args:
- slice_dims: Tuple of two Symbols among (:LL,:LR,:RL,:RR) indicating which dims vary
- slice_idx: Optional Tuple giving fixed indices for the other two dims
- interval: frame interval in milliseconds
- surface3d: whether to use a 3D surface plot
"""
function animate_value_slice(V, grid;
                             slice_dims = (:RL, :RR),
                             slice_idx = nothing,
                             interval = 200,
                             surface3d = false)

    # Map each symbol to its dimension index in V
    dim_map = Dict(:LL=>1, :LR=>2, :RL=>3, :RR=>4)
    all_dims = [:LL, :LR, :RL, :RR]
    # Determine fixed dims & their indices
    fixed_dims = setdiff(all_dims, slice_dims)
    if slice_idx === nothing
        slice_idx = ntuple(i->div(size(V, dim_map[fixed_dims[i]]),2), 2)
    end

    # Build animation
    fps = round(Int, 1000/interval)
    anim = @animate for t in 1:size(V,5)
        # Extract the 2D slice at time t
        inds = [Colon(), Colon(), Colon(), Colon(), t]  # Use array instead of tuple
        # fill in fixed dims
        inds[dim_map[fixed_dims[1]]] = slice_idx[1]
        inds[dim_map[fixed_dims[2]]] = slice_idx[2]
        mat = view(V, inds...)  # this is a zs×zs matrix

        if surface3d
            surface(grid, grid, mat;
                    xlabel=string(slice_dims[1]),
                    ylabel=string(slice_dims[2]),
                    zlabel="V",
                    title="t = $t")
        else
            heatmap(grid, grid, mat;
                    xlabel=string(slice_dims[1]),
                    ylabel=string(slice_dims[2]),
                    title="Value slice at t = $t")
        end
    end

    gif(anim, "value_slice.gif", fps=fps)
end

"""
Animate a 2D slice of the policy D over time.

Arguments:
- D: 5D Array of UInt8 (LL, LR, RL, RR, t), values 1=wait,2=left,3=right
- grid: 1D range of z values
Keyword args:
- slice_dims, slice_idx, interval: as above
"""
# function animate_policy_slice(D, grid;
#                               slice_dims = (:RL, :RR),
#                               slice_idx = nothing,
#                               interval = 200)

#     dim_map = Dict(:LL=>1, :LR=>2, :RL=>3, :RR=>4)
#     all_dims = [:LL, :LR, :RL, :RR]
#     fixed_dims = setdiff(all_dims, slice_dims)
#     if slice_idx === nothing
#         slice_idx = ntuple(i->div(size(D, dim_map[fixed_dims[i]]),2), 2)
#     end

#     fps = round(Int, 1000/interval)
#     anim = @animate for t in 1:size(D,5)
#         inds = [Colon(), Colon(), Colon(), Colon(), t]  # Use array instead of tuple
#         inds[dim_map[fixed_dims[1]]] = slice_idx[1]
#         inds[dim_map[fixed_dims[2]]] = slice_idx[2]
#         mat = view(D, inds...)

#         heatmap(z_grid, z_grid, Float64.(mat);
#                 xlabel=string(slice_dims[1]),
#                 ylabel=string(slice_dims[2]),
#                 title="Policy slice at t = $t (1=Wait, 2=Left, 3=Right)",
#                 c = [:white, :blue, :red],
#                 clims=(0.5,3.5))
#     end

#     gif(anim, "policy_slice.gif", fps=fps)
# end

function animate_policy_slice(D, grid;
    slice_dims = (:RL, :RR),
    slice_idx = nothing,
    interval = 200)

    # map symbols to dims
    dim_map   = Dict(:LL=>1, :LR=>2, :RL=>3, :RR=>4)
    all_dims  = [:LL, :LR, :RL, :RR]
    fixed_dims = setdiff(all_dims, slice_dims)

    # center slice if not given
    if slice_idx === nothing
        slice_idx = ntuple(i->div(size(D, dim_map[fixed_dims[i]]),2), 2)
    end

    fps = round(Int, 1000/interval)

    anim = @animate for t in 1:size(D,5)
        # build view indices
        inds = [Colon(), Colon(), Colon(), Colon(), t]
        inds[dim_map[fixed_dims[1]]] = slice_idx[1]
        inds[dim_map[fixed_dims[2]]] = slice_idx[2]

        mat = Float64.(view(D, inds...))

        heatmap(
            grid, grid, mat;
            xlabel            = string(slice_dims[1]),
            ylabel            = string(slice_dims[2]),
            title             = "Policy @ t = $t",
            aspect_ratio      = 1,
            size              = (500, 500),
            color             = [:white, :blue, :red],
            clims             = (0.5, 3.5),
            colorbar          = true,
            colorbar_ticks    = [1, 2, 3],
            colorbar_tick_labels = ["Wait", "Left", "Right"],
            cbar_title        = "Action",
            legend            = false
        )
    end

    gif(anim, "policy_slice.gif", fps=fps)
end
 
# --- Example usage ---
# Define a node
base_node = Node(0.0, 1.0, 1.0)  # mean_r, var_r, var_x
# Bind var_trans to this node
# var_trans_func = (t, dt) -> var_trans(base_node, t, dt)
# # Create params and run

# Order: zs, ts, z_max, dt, rho_L, rho_R, cost, t_null, tol, priors
p = Stage1Params(50, 100, 3.0, 0.1, 0.5, 0.5, 0.8, 0.2, 1e-4, base_node)

# V, D, z_grid = solve_stage1(p)

# # Save results to JLD2 file
# using JLD2
# @save "stage1_new.jld2" V D z_grid

# To load the results later, use:
# @load "stage1_results.jld2" V D z_grid


# animate_value_slice(V, z_grid; slice_dims=(:LL,:LR), interval=100, surface3d=true)
# animate_policy_slice(D, z_grid; slice_dims=(:RL,:RR), interval=150)

using JLD2, Plots

# # # # 2a) Load
@load "stage1_new.jld2" D z_grid

# # # 2b) get sizes
# # _,_,_,_,T = size(D)
# # N4 = prod(size(D)[1:4])   # total number of grid‐points

# # # 2c) compute fractions
# # fractions = zeros(3, T)   # rows = action 1,2,3; cols = time

# # for t in 1:T
# #     A = D[:,:,:,:,t]
# #     fractions[1,t] = count(==(1), A) / N4   # wait
# #     fractions[2,t] = count(==(2), A) / N4   # choose left
# #     fractions[3,t] = count(==(3), A) / N4   # choose right
# # end

# # # 2d) plot
# # plot(1:T, fractions',
# #      xlabel="time‐step t",
# #      ylabel="fraction of grid‐points",
# #      label=["Wait" "Left" "Right"],
# #      legend=:right)

# zs = length(z_grid)
# println(zs)
# mid = div(zs+1, 2)

# # # 2) 选几个 t 做对比
# # ts = [1, Int(round(size(D,5)/2)), size(D,5)]  # 比如 t=1, 中期, 末期

# # # 3) 生成子图
# # plots = Vector{Plots.Plot}(undef, length(ts))
# # for (i,t) in enumerate(ts)
# #     # 取出切片：fix RL=mid, RR=mid，只保留 LL×LR
# #     mat = view(D, :, :, 3, 5, t)
# #     # 转成 Float (1,2,3)
# #     matf = Float64.(mat)
# #     plots[i] = heatmap(
# #         z_grid, z_grid, matf;
# #         xlabel="z_LL", ylabel="z_LR",
# #         title="Policy @ t=$(t)",
# #         clims=(0.5, 3.5),           # 只出现一次 clims
# #         colorbar_tickfont=font(8),
# #         legend=false,
# #         c = [:white, :blue, :red]  # white=wait, blue=left, red=right
# #       )
# # end

# # plot(plots...; layout=(1,length(ts)), size=(300*length(ts),300))

animate_policy_slice(D, z_grid;
                    slice_dims = (:LR,:RL),
                    slice_idx  = (50,40),
                    interval   = 100)