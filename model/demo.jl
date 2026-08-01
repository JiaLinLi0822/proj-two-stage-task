using Random
using Statistics
using Distributions
using PyPlot
using PyCall
using MvNormalCDF

# ---------------- utils ----------------
normalize_prob(p::AbstractVector{<:Real}) = (s = sum(p); s>0 ? p ./ s : fill(1/length(p), length(p)))
function drawcat(p::AbstractVector{<:Real})
    q = normalize_prob(collect(p)); r = rand(); acc = 0.0
    @inbounds for (i,pi) in enumerate(q); acc += pi; if r ≤ acc; return i; end; end
    return length(q)
end
entropy(p::AbstractVector{<:Real}) = begin
    q = normalize_prob(collect(p)); s = 0.0
    @inbounds for pi in q
        if pi > 0; s -= pi*log(pi); end
    end
    s
end

# --------------- tree ------------------
struct Tree2LayerFull
    nnodes::Int
    paths::Matrix{Int}              # (4, 2) actions ∈ {1,2}
    path_nodes::Vector{Vector{Int}} # nodes per path (indices 1..6)
    depth::Vector{Int}
end

function make_tree_2layer_full()
    paths = [1 1; 1 2; 2 1; 2 2]            # [LL, LR, RL, RR]
    path_nodes = [[1,3],[1,4],[2,5],[2,6]]  # 1:L,2:R,3:LL,4:LR,5:RL,6:RR
    depth = [1,1,2,2,2,2]
    Tree2LayerFull(6, paths, path_nodes, depth)
end

# -------- node reward posteriors (Normal–Normal) --------
mutable struct NodeBeliefs
    mu::Vector{Float64}
    var::Vector{Float64}
    mu0::Float64
    tau0sq::Float64
    sigmasq::Float64
    sumx::Vector{Float64}
    cnt::Vector{Int}
end

function NodeBeliefs(nnodes; mu0=0.0, tau0=1.0, sigma=0.6)
    NodeBeliefs(fill(mu0, nnodes), fill(tau0^2, nnodes), mu0, tau0^2, sigma^2,
                zeros(nnodes), zeros(Int, nnodes))
end

@inline function posterior_params(mu0, tau0sq, sigmasq, sumx, cnt)
    τpostsq = 1.0 / (1/tau0sq + cnt/sigmasq)
    μpost   = τpostsq * (mu0/tau0sq + sumx/sigmasq)
    μpost, τpostsq
end

# ------------ path incidence (depth-discounted) ----------
function path_incidence_matrix(tree::Tree2LayerFull; γ=1.0)
    nπ, nn = size(tree.paths,1), tree.nnodes
    A = zeros(Float64, nπ, nn)
    for p in 1:nπ
        for i in tree.path_nodes[p]
            A[p,i] = γ^(tree.depth[i]-1)
        end
    end
    A
end


function policy_likelihood_optimal_Genz(tree::Tree2LayerFull, nb::NodeBeliefs; γ=1.0, m::Int=5000)
    nπ = size(tree.paths, 1)
    A  = path_incidence_matrix(tree; γ=γ)

    # 节点后验：R ~ N(mu_R, Σ_R)
    μR = nb.mu
    ΣR = Diagonal(nb.var)          # 6×6

    # 路径价值：V = A*R  =>  V ~ N(μV, ΣV)
    μV = A * μR                    # 4×1
    ΣV = A * ΣR * A'               # 4×4

    lik = zeros(Float64, nπ)

    for k in 1:nπ
        # 构造差值矩阵 D_k: 每行是 e_k - e_j, j ≠ k
        others = setdiff(1:nπ, [k])
        D = zeros(Float64, length(others), nπ)
        for (row, j) in enumerate(others)
            D[row, k] =  1.0
            D[row, j] = -1.0
        end

        μY = D * μV                 # 3×1
        ΣY = D * ΣV * D'            # 3×3

        a = zeros(length(others))   # 下界 0
        b = fill(Inf, length(others))

        # Genz / quasi-MC 计算 P( Y ∈ [0,∞)^3 )
        (p, err) = mvnormcdf(μY, ΣY, a, b; m=m)
        lik[k] = p
    end

    lik .+= 1e-12
    lik ./= sum(lik)
    return lik
end

# ------ policy likelihood: P(π is optimal) via MC -------
function policy_likelihood_optimal_MC(tree::Tree2LayerFull, nb::NodeBeliefs; γ=1.0, nsamples=4000)
    nπ = size(tree.paths,1)
    A  = path_incidence_matrix(tree; γ=γ)
    counts = zeros(Int, nπ)
    σ = sqrt.(max.(nb.var, 1e-12))
    for _ in 1:nsamples
        Rdraw = rand.(Normal.(nb.mu, σ))          # draw node values
        V = A * collect(Rdraw)                    # path values
        pwin = findmax(V)[2]                      # winner path
        counts[pwin] += 1
    end
    lik = counts ./ max(nsamples,1)
    lik = (lik .+ 1e-12); lik ./= sum(lik)
    lik
end

function feasible_nodes(tree::Tree2LayerFull, feasible::Vector{Int})
    allow = falses(tree.nnodes)
    for p in feasible
        for i in tree.path_nodes[p]
            allow[i] = true
        end
    end
    findall(allow)
end

# —— 计算节点之间的无向图最短距离（父子连边）——
function node_distance_matrix(tree::Tree2LayerFull)
    nn = tree.nnodes
    adj = [Int[] for _ in 1:nn]
    # 父子连边：1↔3, 1↔4, 2↔5, 2↔6
    push!(adj[1], 3, 4); push!(adj[3], 1); push!(adj[4], 1)
    push!(adj[2], 5, 6); push!(adj[5], 2); push!(adj[6], 2)
    # 如需认为顶层左右(1,2)也相对“临近”，可解除下一行注释，使 dist(1,2)=1
    # push!(adj[1], 2); push!(adj[2], 1)

    D = fill(typemax(Int), nn, nn)
    for i in 1:nn
        D[i,i] = 0
        # BFS
        dist = fill(-1, nn); dist[i] = 0
        q = [i]
        while !isempty(q)
            u = popfirst!(q)
            for v in adj[u]
                if dist[v] == -1
                    dist[v] = dist[u] + 1
                    push!(q, v)
                end
            end
        end
        for j in 1:nn
            if dist[j] != -1
                D[i,j] = dist[j]
            end
        end
    end
    D
end

function sample_node_persist_prox(
    tree::Tree2LayerFull,
    Π::Vector{Float64},
    feasible::Vector{Int};
    allowed_nodes::Union{Vector{Int},Nothing}=nothing,
    last_node::Union{Int,Nothing}=nothing,
    D::Union{Matrix{Int},Nothing}=nothing,
    p_stay::Float64=0.7,
    λ::Float64=1.0,
    ε::Float64=0.05
)
    nn = tree.nnodes
    # 若未给出，默认允许所有节点（兼容 Stage 1）
    allow = isnothing(allowed_nodes) ? collect(1:nn) : allowed_nodes
    allow_mask = falses(nn); allow_mask[allow] .= true

    # —— 策略→基础分布（先算，再掩蔽到 allow）——
    Πf = normalize_prob([Π[p] for p in feasible])
    p_base = zeros(Float64, nn)
    for (j,p) in enumerate(feasible)
        for i in tree.path_nodes[p]
            p_base[i] += Πf[j] / length(tree.path_nodes[p])
        end
    end
    p_base .= allow_mask .* p_base
    if sum(p_base) > 0
        p_base ./= sum(p_base)
    else
        # 退化情形：在允许集合里均匀
        p_base .= 0.0; p_base[allow] .= 1/length(allow)
    end
    # 只在允许集合里做 ε 探索
    p_base = normalize_prob((1-ε).*p_base .+ ε/length(allow) .* allow_mask)

    # —— 若没有 last_node：直接从允许集合的 p_base 取样 —— 
    if last_node === nothing || !allow_mask[last_node]
        # 若 last_node 不在允许集合，强制 p_stay=0（等价于上式）
        return drawcat(p_base)
    end

    Dm = isnothing(D) ? node_distance_matrix(tree) : D

    # —— 切换核 k：只在允许集合、且不等于 last_node 上有质量 —— 
    k = zeros(Float64, nn)
    for i in allow
        if i != last_node && Dm[last_node, i] < typemax(Int)
            k[i] = exp(-λ * Dm[last_node, i])
        end
    end
    if sum(k) > 0
        k ./= sum(k)
    else
        # 无近邻时，在允许集合（排除 last_node）均匀
        for i in allow
            if i != last_node; k[i] = 1.0; end
        end
        k ./= sum(k)
    end

    # —— 合成最终分布（完全掩蔽到 allow）——
    pswitch_raw = p_base .* k
    pswitch = sum(pswitch_raw) > 0 ? pswitch_raw ./ sum(pswitch_raw) : k

    p_final = zeros(Float64, nn)
    p_final[last_node] = p_stay
    p_final .+= (1 - p_stay) .* pswitch

    # 仅在允许集合内做 ε 探索并归一化
    p_final .= allow_mask .* p_final
    if sum(p_final) == 0
        p_final .= 0.0; p_final[allow] .= 1/length(allow)
    end
    p_final = normalize_prob((1-ε).*p_final .+ ε/length(allow) .* allow_mask)

    return drawcat(p_final)
end

# ------ node selection from a policy posterior Π (empirical) --------
# Restrict to feasible policies; distribute mass to their nodes (with ε-explore).
function sample_node_from_policy_posterior(tree::Tree2LayerFull, Π::Vector{Float64},
                                           feasible::Vector{Int}; ε::Float64=0.05)
    nn = tree.nnodes
    pnode = fill(ε/nn, nn)                     # small exploration
    Πf = zeros(length(feasible))
    for (j,p) in enumerate(feasible); Πf[j] = Π[p]; end
    Πf = normalize_prob(Πf)
    for (j,p) in enumerate(feasible)
        for i in tree.path_nodes[p]
            pnode[i] += (1.0 - ε) * Πf[j] / length(tree.path_nodes[p])
        end
    end
    drawcat(pnode)
end

# ------ compute policy posterior from likelihood with temperature ------
function posterior_over_paths(lik::Vector{Float64}; Tsoft::Float64=1.0)
    """
    Compute policy posterior from likelihood using temperature.
    Posterior ∝ exp(log(lik) / Tsoft)
    
    Parameters:
    - lik: policy likelihood from MC sampling
    - Tsoft: temperature parameter (higher = more uniform)
    """
    logu = log.(lik .+ 1e-12) ./ max(Tsoft, 1e-12)
    u = exp.(logu .- maximum(logu)) # stabilize
    normalize_prob(u)
end

# ------ posterior update with time step Δt ------
function update_posterior_with_timestep(Π_current::Vector{Float64}, Π_target::Vector{Float64};
                                       Δt::Float64=0.001, α::Float64=1.0)
    """
    Update posterior using time step Δt.
    Π(t+Δt) = Π(t) + Δt * α * (Π_target - Π(t))
    
    Parameters:
    - Π_current: current posterior
    - Π_target: target posterior (from likelihood computation)
    - Δt: time step (default 0.001s)
    - α: update rate constant (default 1.0)
    """
    Π_new = Π_current .+ Δt * α .* (Π_target .- Π_current)
    normalize_prob(Π_new)
end

# ------ Main: two-stage loop (NO MH; direct posterior from MC likelihood) ------
function run_trial_two_stage(; rngseed=0,
    mu0=0.0, tau0=1.0, sigma=0.6,     # evidence noise
    γ=1.0,                     # depth discount & optional utility bias
    nsamples_lik=3000,                # MC for P(optimal)
    H_thresh1=log(4),            # entropy thresholds on feasible
    H_thresh2=log(2),
    Tsoft=1.0,
    eps_nodesel=0.05,
    Δt=0.01,                         # time step for posterior updating (seconds)
    α=1.0,                            # update rate constant for time-stepping
    R_true = [1.0, 0.0, 3.0, 1.0, 4.0, 2.0]  # true rewards (L,R,LL,LR,RL,RR)
)

    Random.seed!(rngseed)
    tree = make_tree_2layer_full()
    nπ   = size(tree.paths,1)
    nb   = NodeBeliefs(tree.nnodes; mu0=mu0, tau0=tau0, sigma=sigma)

    # ---------------- Stage 1 ----------------
    feasible1 = collect(1:nπ)
    nodes_obs_stage1 = Int[]
    RT_sub_stage1 = Int[]  # here each re-evaluation counts as 1 "step"
    policy_posteriors_stage1 = Vector{Vector{Float64}}()
    reward_estimates_stage1 = Vector{Vector{Float64}}()
    reward_variances_stage1 = Vector{Vector{Float64}}()

    # initial policy posterior: uniform on feasible
    Π = zeros(nπ); Π[feasible1] .= 1/length(feasible1)
    push!(policy_posteriors_stage1, copy(Π))
    push!(reward_estimates_stage1, copy(nb.mu))
    push!(reward_variances_stage1, copy(nb.var))

    D = node_distance_matrix(tree)
    last_node = nothing
    while true
        # choose a node to observe using current Π

        node = sample_node_persist_prox(
            tree, Π, feasible1;
            last_node=last_node, D=D,
            p_stay=0.75, λ=1.2, ε=eps_nodesel
        )
        last_node = node   
        push!(nodes_obs_stage1, node)

        # observe & update node posterior (Normal–Normal)
        x = rand(Normal(R_true[node], sigma))
        nb.sumx[node] += x; nb.cnt[node] += 1
        nb.mu[node], nb.var[node] = posterior_params(nb.mu0, nb.tau0sq, nb.sigmasq,
                                                     nb.sumx[node], nb.cnt[node])

        # recompute policy likelihood via MC
        # lik = policy_likelihood_optimal_MC(tree, nb; γ=γ, nsamples=nsamples_lik)
        lik = policy_likelihood_optimal_Genz(tree, nb; γ=γ)

        # compute target posterior from likelihood with temperature
        Π_target = posterior_over_paths(lik; Tsoft=Tsoft)
        
        # update posterior with time step Δt
        Π = update_posterior_with_timestep(Π, Π_target; Δt=Δt, α=α)

        push!(RT_sub_stage1, 1)
        push!(policy_posteriors_stage1, copy(Π))
        push!(reward_estimates_stage1, copy(nb.mu))
        push!(reward_variances_stage1, copy(nb.var))

        Πf = normalize_prob([Π[p] for p in feasible1])
        if entropy(Πf) ≤ H_thresh1
            break
        end
    end
    π1 = argmax(Π)
    a1 = tree.paths[π1, 1]

    # ---------------- Stage 2 ----------------
    feasible2 = [p for p in feasible1 if tree.paths[p,1] == a1]
    isempty(feasible2) && (feasible2 = [p for p in 1:nπ if tree.paths[p,1] == a1])

    allowed2 = feasible_nodes(tree, feasible2)   # 只允许该分支上的节点

    nodes_obs_stage2 = Int[]
    RT_sub_stage2 = Int[]
    policy_posteriors_stage2 = Vector{Vector{Float64}}()
    reward_estimates_stage2 = Vector{Vector{Float64}}()
    reward_variances_stage2 = Vector{Vector{Float64}}()

    Π2 = zeros(nπ); Π2[feasible2] .= 1/length(feasible2)
    push!(policy_posteriors_stage2, copy(Π2))
    push!(reward_estimates_stage2, copy(nb.mu))
    push!(reward_variances_stage2, copy(nb.var))

    while true

        node = sample_node_persist_prox(
            tree, Π2, feasible2;
            allowed_nodes=allowed2,        
            last_node=last_node, D=D,
            p_stay=0.75, λ=1.2, ε=eps_nodesel
        )
        last_node = node

        push!(nodes_obs_stage2, node)

        x = rand(Normal(R_true[node], sigma))
        nb.sumx[node] += x; nb.cnt[node] += 1
        nb.mu[node], nb.var[node] = posterior_params(nb.mu0, nb.tau0sq, nb.sigmasq,
                                                     nb.sumx[node], nb.cnt[node])

        lik2 = policy_likelihood_optimal_MC(tree, nb; γ=γ, nsamples=nsamples_lik)

        # compute target posterior from likelihood with temperature
        Π_all_target = posterior_over_paths(lik2; Tsoft=Tsoft)
        
        # restrict target to feasible2
        Π2_target = zeros(nπ)
        Π2_target[feasible2] = normalize_prob([Π_all_target[p] for p in feasible2])
        
        # update posterior with time step Δt
        Π2 = update_posterior_with_timestep(Π2, Π2_target; Δt=Δt, α=α)

        push!(RT_sub_stage2, 1)
        push!(policy_posteriors_stage2, copy(Π2))
        push!(reward_estimates_stage2, copy(nb.mu))
        push!(reward_variances_stage2, copy(nb.var))

        Πf2 = normalize_prob([Π2[p] for p in feasible2])
        if entropy(Πf2) ≤ H_thresh2
            break
        end
    end
    π2 = argmax(Π2)
    a2 = tree.paths[π2, 2]

    leaf_names = ["LL","LR","RL","RR"]
    chosen_idx = findfirst(p -> (tree.paths[p,1]==a1 && tree.paths[p,2]==a2), 1:nπ)

    return (; 
        actions = [a1,a2],
        first_stage_RT = length(RT_sub_stage1),
        second_stage_RT = length(RT_sub_stage2),
        chosen_path = leaf_names[chosen_idx],
        observed_nodes_stage1 = nodes_obs_stage1,
        observed_nodes_stage2 = nodes_obs_stage2,
        RT_sub_stage1 = RT_sub_stage1,
        RT_sub_stage2 = RT_sub_stage2,
        policy_posteriors_stage1 = policy_posteriors_stage1,
        policy_posteriors_stage2 = policy_posteriors_stage2,
        reward_estimates_stage1 = reward_estimates_stage1,
        reward_estimates_stage2 = reward_estimates_stage2,
        reward_variances_stage1 = reward_variances_stage1,
        reward_variances_stage2 = reward_variances_stage2,
        mu_post = copy(nb.mu),
        var_post = copy(nb.var),
        R_true = R_true
    )
end

# -------- example run --------
res = run_trial_two_stage(rngseed=69,
                          sigma=1.0, γ=1.0,
                          nsamples_lik=1000,
                          H_thresh1=1.2,
                          H_thresh2=0.65,
                          Tsoft=1.0,
                          eps_nodesel=0.05)

println("Actions: ", res.actions,
        "  RT1=", res.first_stage_RT, "  RT2=", res.second_stage_RT,
        "  Path=", res.chosen_path)

println("\nStage 1:")
for (i, (node, rt)) in enumerate(zip(res.observed_nodes_stage1, res.RT_sub_stage1))
    println("  Observation $i: node=$node, step=$rt")
end

println("\nStage 2:")
for (i, (node, rt)) in enumerate(zip(res.observed_nodes_stage2, res.RT_sub_stage2))
    println("  Observation $i: node=$node, step=$rt")
end

# Print posterior distributions for each reward node
node_names = ["L", "R", "LL", "LR", "RL", "RR"]
println("\nPosterior distributions for reward nodes:")
for i in 1:length(res.mu_post)
    μ = res.mu_post[i]
    σ = sqrt(res.var_post[i])
    println("  Node $i ($(node_names[i])): μ = $(round(μ, digits=3)), σ = $(round(σ, digits=3))  [N($(round(μ, digits=3)), $(round(σ^2, digits=3)))]")
end

# -------- Helper function to draw a single frame --------
function draw_frame(res, tree, frame_info; save_path=nothing)
    node_names = ["L", "R", "LL", "LR", "RL", "RR"]
    path_names = ["LL", "LR", "RL", "RR"]
    
    fig = figure(figsize=(18, 12))
    
    # Use subplot2grid for better PyCall compatibility
    # Main tree plot (top, spans 2 columns, row 0)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1, fig=fig)
    ax1.axis("off")
    ax1.set_aspect("equal", adjustable="box")
    
    # Tree coordinates
    coords = Dict(
        0 => (0.0, 0.0),
        1 => (-1.0, -1.0),     # L
        2 => (1.0, -1.0),      # R
        3 => (-1.5, -2.0),     # LL
        4 => (-0.5, -2.0),     # LR
        5 => (0.5, -2.0),       # RL
        6 => (1.5, -2.0)        # RR
    )
    
    # Draw root node
    root_circle = plt.Circle((0, 0), 0.12, color="white", ec="black", linewidth=2, zorder=4)
    ax1.add_patch(root_circle)
    ax1.text(0, 0, "Root", ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)
    
    # Determine chosen path for highlighting
    chosen_path_idx = findfirst(p -> (tree.paths[p,1]==res.actions[1] && tree.paths[p,2]==res.actions[2]), 1:4)
    chosen_nodes = tree.path_nodes[chosen_path_idx]
    
    # Draw tree edges
    if 1 in chosen_nodes
        ax1.plot([0, -1], [0, -1], "g-", linewidth=3, alpha=0.7, label="Chosen path")
    else
        ax1.plot([0, -1], [0, -1], "k-", linewidth=2, label="Left")
    end
    if 2 in chosen_nodes
        ax1.plot([0, 1], [0, -1], "g-", linewidth=3, alpha=0.7)
    else
        ax1.plot([0, 1], [0, -1], "k-", linewidth=2, label="Right")
    end
    # L to LL and LR
    if 3 in chosen_nodes
        ax1.plot([-1, -1.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([-1, -1.5], [-1, -2], "k-", linewidth=1.5)
    end
    if 4 in chosen_nodes
        ax1.plot([-1, -0.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([-1, -0.5], [-1, -2], "k-", linewidth=1.5)
    end
    # R to RL and RR
    if 5 in chosen_nodes
        ax1.plot([1, 0.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([1, 0.5], [-1, -2], "k-", linewidth=1.5)
    end
    if 6 in chosen_nodes
        ax1.plot([1, 1.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([1, 1.5], [-1, -2], "k-", linewidth=1.5)
    end
    
    # Draw nodes with rewards
    for (node_idx, (x, y)) in coords
        if node_idx > 0
            is_chosen_leaf = (node_idx in chosen_nodes && node_idx >= 3)
            node_color = is_chosen_leaf ? "lightgreen" : "lightblue"
            node_ec = is_chosen_leaf ? "green" : "black"
            node_lw = is_chosen_leaf ? 3 : 2
            
            circle = plt.Circle((x, y), 0.15, color=node_color, ec=node_ec, linewidth=node_lw, zorder=3)
            ax1.add_patch(circle)
            ax1.text(x, y + 0.35, node_names[node_idx], ha="center", fontsize=12, fontweight="bold")
            r_val = res.R_true[node_idx]
            ax1.text(x, y - 0.35, "R=$(round(r_val, digits=2))", ha="center", fontsize=10, 
                    color="red", fontweight="bold")
        end
    end
    
    # Mark only the current fixated node
    stage = frame_info[:stage]
    obs_idx = frame_info[:obs_idx]
    current_fixated_node = nothing
    
    if stage == 1
        if obs_idx > 0
            # Only mark the current fixated node
            current_fixated_node = res.observed_nodes_stage1[obs_idx]
            if current_fixated_node in keys(coords)
                x, y = coords[current_fixated_node]
                circle = plt.Circle((x, y), 0.2, color="yellow", alpha=0.7, ec="orange", 
                                  linewidth=4, zorder=2)
                ax1.add_patch(circle)
            end
        end
    elseif stage == 2
        if obs_idx > 0
            # Only mark the current fixated node in stage 2
            current_fixated_node = res.observed_nodes_stage2[obs_idx]
            if current_fixated_node in keys(coords)
                x, y = coords[current_fixated_node]
                circle = plt.Circle((x, y), 0.2, color="lightgreen", alpha=0.7, ec="green", 
                                  linewidth=4, zorder=2)
                ax1.add_patch(circle)
            end
        end
    end
    
    # Display fixation node and policy posteriors on the right side of the tree
    if stage == 1 && obs_idx > 0
        Π = res.policy_posteriors_stage1[obs_idx+1]
        fixated_node_name = node_names[res.observed_nodes_stage1[obs_idx]]
        
        # Build text with new format
        info_text = "Fixation node: $fixated_node_name\n\n"
        for (p_idx, p_name) in enumerate(path_names)
            info_text *= "Path $p_name: $(round(Π[p_idx], digits=3))\n"
        end
        
        ax1.text(2.5, 0, info_text, ha="left", va="center", fontsize=11,
                transform=ax1.transData, family="monospace")
    elseif stage == 2 && obs_idx > 0
        Π = res.policy_posteriors_stage2[obs_idx+1]
        fixated_node_name = node_names[res.observed_nodes_stage2[obs_idx]]
        
        # Build text with new format
        info_text = "Fixation node: $fixated_node_name\n\n"
        for (p_idx, p_name) in enumerate(path_names)
            if Π[p_idx] > 0.01
                info_text *= "Path $p_name: $(round(Π[p_idx], digits=3))\n"
            end
        end
        
        ax1.text(2.5, 0, info_text, ha="left", va="center", fontsize=11,
                transform=ax1.transData, family="monospace")
    elseif obs_idx == 0
        # Initial state
        info_text = "Fixation node: None\n\n"
        if stage == 1
            Π = res.policy_posteriors_stage1[1]
        else
            Π = res.policy_posteriors_stage2[1]
        end
        for (p_idx, p_name) in enumerate(path_names)
            info_text *= "Path $p_name: $(round(Π[p_idx], digits=3))\n"
        end
        ax1.text(2.5, 0, info_text, ha="left", va="center", fontsize=11,
                transform=ax1.transData, family="monospace")
    end
    
    ax1.set_xlim(-2.2, 4.0)  # Extended to accommodate text on the right
    ax1.set_ylim(-2.8, 0.6)
    title_str = "Decision Tree - "
    if stage == 1
        if obs_idx == 0
            title_str *= "Stage 1, Initial State"
        else
            title_str *= "Stage 1, Observation $obs_idx"
        end
    else
        if obs_idx == 0
            title_str *= "Stage 2, Initial State"
        else
            title_str *= "Stage 2, Observation $obs_idx"
        end
    end
    ax1.set_title(title_str, fontsize=14, fontweight="bold")
    
    # Policy posterior line chart (middle left)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=1, fig=fig)
    path_names = ["LL", "LR", "RL", "RR"]
    path_colors = ["blue", "red", "green", "orange"]
    
    if stage == 1
        n_obs = length(res.observed_nodes_stage1)
        max_idx = min(obs_idx, n_obs)
        x_pos = collect(0:max_idx)
        
        if max_idx >= 0 && length(res.policy_posteriors_stage1) > 0
            for (path_idx, path_name) in enumerate(path_names)
                n_posteriors = min(max_idx+1, length(res.policy_posteriors_stage1))
                if n_posteriors > 0
                    posteriors = [Π[path_idx] for Π in res.policy_posteriors_stage1[1:n_posteriors]]
                    if length(posteriors) == length(x_pos)
                        ax2.plot(x_pos, posteriors, "o-", label=path_name, alpha=0.8, 
                                linewidth=2, markersize=6, color=path_colors[path_idx])
                    end
                end
            end
        end
        
        ax2.set_xlabel("Time Step", fontsize=11)
        ax2.set_ylabel("Policy Posterior", fontsize=11)
        ax2.set_title("Stage 1: Policy Posterior Evolution", fontsize=12, fontweight="bold")
        if max_idx >= 0
            # Show time steps, with intervals if too many
            if length(x_pos) <= 10
                # Show all time steps
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([string(i) for i in x_pos], fontsize=9)
            else
                # Show time steps with intervals
                step = max(1, div(length(x_pos), 10))
                tick_positions = collect(0:step:max_idx)
                ax2.set_xticks(tick_positions)
                ax2.set_xticklabels([string(i) for i in tick_positions], fontsize=9)
            end
        end
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1)
    else
        ax2.text(0.5, 0.5, "Stage 1 Complete", ha="center", va="center", 
                transform=ax2.transAxes, fontsize=14, fontweight="bold")
        ax2.axis("off")
    end
    
    # Stage 2 line chart (middle right)
    ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=1, rowspan=1, fig=fig)
    if stage == 2
        n_obs = length(res.observed_nodes_stage2)
        max_idx = min(obs_idx, n_obs)
        x_pos2 = collect(0:max_idx)
        
        feasible_paths = [p for p in 1:4 if any(Π[p] > 0.01 for Π in res.policy_posteriors_stage2)]
        
        for (feas_idx, path_idx) in enumerate(feasible_paths)
            n_posteriors = min(max_idx+1, length(res.policy_posteriors_stage2))
            posteriors = [Π[path_idx] for Π in res.policy_posteriors_stage2[1:n_posteriors]]
            if length(posteriors) == length(x_pos2)
                ax3.plot(x_pos2, posteriors, "o-", label=path_names[path_idx], alpha=0.8,
                        linewidth=2, markersize=6, color=path_colors[path_idx])
            end
        end
        
        ax3.set_xlabel("Time Step", fontsize=11)
        ax3.set_ylabel("Policy Posterior", fontsize=11)
        ax3.set_title("Stage 2: Policy Posterior Evolution", fontsize=12, fontweight="bold")
        # Show time steps, with intervals if too many
        if length(x_pos2) <= 10
            # Show all time steps
            ax3.set_xticks(x_pos2)
            ax3.set_xticklabels([string(i) for i in x_pos2], fontsize=9)
        else
            # Show time steps with intervals
            step = max(1, div(length(x_pos2), 10))
            tick_positions = collect(0:step:max_idx)
            ax3.set_xticks(tick_positions)
            ax3.set_xticklabels([string(i) for i in tick_positions], fontsize=9)
        end
        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.set_ylim(0, 1)
    else
        ax3.text(0.5, 0.5, "Stage 2\n(Not started)", ha="center", va="center", 
                transform=ax3.transAxes, fontsize=12)
        ax3.axis("off")
    end
    
    # Reward estimate as Gaussian distributions (bottom, spans 2 columns)
    # Shows only current state - one frame per observation
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2, rowspan=1, fig=fig)
    node_names_plot = ["L", "R", "LL", "LR", "RL", "RR"]
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    
    # Get current estimates (only the latest state, not history)
    if stage == 1
        if obs_idx == 0
            # Initial state
            current_estimates = res.reward_estimates_stage1[1]
            current_variances = res.reward_variances_stage1[1]
        else
            # After obs_idx observations in stage 1
            current_estimates = res.reward_estimates_stage1[obs_idx+1]
            current_variances = res.reward_variances_stage1[obs_idx+1]
        end
    else
        # Stage 2
        if obs_idx == 0
            # Initial state of stage 2
            current_estimates = res.reward_estimates_stage2[1]
            current_variances = res.reward_variances_stage2[1]
        else
            # After obs_idx observations in stage 2
            current_estimates = res.reward_estimates_stage2[obs_idx+1]
            current_variances = res.reward_variances_stage2[obs_idx+1]
        end
    end
    
    # Determine reward value range for plotting
    stds = [sqrt(max(1e-6, current_variances[i])) for i in 1:6]
    min_reward = minimum(current_estimates .- 4 .* stds)
    max_reward = maximum(current_estimates .+ 4 .* stds)
    # Also include true rewards
    min_reward = min(min_reward, minimum(res.R_true) - 1)
    max_reward = max(max_reward, maximum(res.R_true) + 1)
    
    # Create fine grid for reward values
    reward_grid = collect(range(min_reward, max_reward, length=300))
    
    # Plot current Gaussian distributions for each node
    for node_idx in 1:6
        μ = current_estimates[node_idx]
        σ = sqrt(max(1e-6, current_variances[node_idx]))  # avoid division by zero
        
        # Compute Gaussian PDF
        pdf_vals = exp.(-0.5 .* ((reward_grid .- μ) ./ σ).^2) ./ (σ * sqrt(2π))
        
        # Plot the Gaussian distribution
        ax4.plot(reward_grid, pdf_vals, "-", color=colors[node_idx], 
                linewidth=2.5, alpha=0.8, label=node_names_plot[node_idx])
        ax4.fill_between(reward_grid, 0, pdf_vals, color=colors[node_idx], alpha=0.3)
        
        # Mark mean with vertical line
        max_pdf = maximum(pdf_vals)
        ax4.plot([μ, μ], [0, max_pdf], "--", color=colors[node_idx], 
                linewidth=1.5, alpha=0.7)
        
        # Mark true reward value with vertical line
        ax4.axvline(x=res.R_true[node_idx], color=colors[node_idx], 
                   linestyle=":", linewidth=2, alpha=0.6)
    end
    
    ax4.set_xlabel("Reward Estimate", fontsize=11)
    ax4.set_ylabel("Probability Density", fontsize=11)
    title_str = "Reward Distributions"
    if stage == 1
        if obs_idx == 0
            title_str *= " - Stage 1 (Initial)"
        else
            title_str *= " - Stage 1 (After Obs $obs_idx)"
        end
    else
        if obs_idx == 0
            title_str *= " - Stage 2 (Initial)"
        else
            title_str *= " - Stage 2 (After Obs $obs_idx)"
        end
    end
    ax4.set_title(title_str, fontsize=12, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8, ncol=3)
    ax4.grid(alpha=0.3, axis="both")
    ax4.set_ylim(bottom=0)  # Start y-axis at 0
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, 
                       hspace=0.35, wspace=0.3)
    
    if save_path !== nothing
        PyPlot.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    end
    
    close(fig)
end

# -------- GIF creation function --------
function create_decision_tree_gif(res, tree; output_path="decision_tree_temporal.gif", frame_duration=1.0, frame_repeat=1)
    node_names = ["L", "R", "LL", "LR", "RL", "RR"]
    
    # Create temporary directory for frames
    temp_dir = mktempdir()
    frame_files = String[]
    
    println("Creating GIF frames...")
    
    # Frame 0: Initial state
    frame_path = joinpath(temp_dir, "frame_000.png")
    draw_frame(res, tree, (stage=1, obs_idx=0), save_path=frame_path)
    push!(frame_files, frame_path)
    
    # Stage 1 frames
    for obs_idx in 1:length(res.observed_nodes_stage1)
        frame_path = joinpath(temp_dir, "frame_$(lpad(obs_idx, 3, "0")).png")
        draw_frame(res, tree, (stage=1, obs_idx=obs_idx), save_path=frame_path)
        push!(frame_files, frame_path)
    end
    
    # Stage 2 frames
    for obs_idx in 1:length(res.observed_nodes_stage2)
        frame_path = joinpath(temp_dir, "frame_$(lpad(length(res.observed_nodes_stage1) + obs_idx, 3, "0")).png")
        draw_frame(res, tree, (stage=2, obs_idx=obs_idx), save_path=frame_path)
        push!(frame_files, frame_path)
    end
    
    # Final frame: show chosen path
    frame_path = joinpath(temp_dir, "frame_final.png")
    draw_frame(res, tree, (stage=2, obs_idx=length(res.observed_nodes_stage2)), save_path=frame_path)
    push!(frame_files, frame_path)
    
    println("Combining frames into GIF...")
    
    # Optionally repeat frames to slow down animation (if duration doesn't work in viewer)
    if frame_repeat > 1
        repeated_files = String[]
        for f in frame_files
            for _ in 1:frame_repeat
                push!(repeated_files, f)
            end
        end
        frame_files = repeated_files
        frame_duration = frame_duration / frame_repeat  # Adjust duration accordingly
    end
    
    # Use Python's PIL/Pillow to create GIF
    try
        # Try using imageio v2 API (newer versions)
        try
            imageio = pyimport("imageio")
            images = [imageio.imread(f) for f in frame_files]
            # Try v2 API first
            try
                imageio.v2.mimsave(output_path, images, duration=frame_duration, loop=0)
            catch
                # Fallback to v1 API
                durations = fill(frame_duration, length(images))
                imageio.mimsave(output_path, images, duration=durations)
            end
        catch
            # Fallback to PIL/Pillow
            PIL = pyimport("PIL.Image")
            images = [PIL.open(f) for f in frame_files]
            if length(images) > 0
                # PIL uses duration in milliseconds
                # Create a list with duration for each frame
                durations_ms = Int(round(frame_duration * 1000))
                # Some viewers ignore duration if not set per frame, so we set it explicitly
                for img in images
                    img.info["duration"] = durations_ms
                end
                images[1].save(output_path, save_all=true, append_images=images[2:end], 
                              duration=durations_ms, loop=0, save_format="GIF")
            end
        end
    catch e
        println("Error creating GIF: $e")
        println("Please install Python packages: pip install imageio pillow")
        rethrow(e)
    end
    
    # Clean up temporary files
    for f in frame_files
        rm(f, force=true)
    end
    rm(temp_dir, force=true)
    
    println("GIF saved as '$output_path'")
end

# -------- Legacy static plot function --------
function plot_decision_tree_temporal(res, tree)
    node_names = ["L", "R", "LL", "LR", "RL", "RR"]
    path_names = ["LL", "LR", "RL", "RR"]
    
    # Create figure with subplots
    fig = figure(figsize=(16, 10))
    
    # Main tree plot
    ax1 = subplot(2, 2, (1, 2))
    ax1.axis("off")
    
    # Tree coordinates
    # Root at (0, 0)
    # Level 1: L at (-1, -1), R at (1, -1)
    # Level 2: LL at (-1.5, -2), LR at (-0.5, -2), RL at (0.5, -2), RR at (1.5, -2)
    coords = Dict(
        0 => (0.0, 0.0),      # root (not shown, but for reference)
        1 => (-1.0, -1.0),     # L
        2 => (1.0, -1.0),      # R
        3 => (-1.5, -2.0),     # LL
        4 => (-0.5, -2.0),     # LR
        5 => (0.5, -2.0),       # RL
        6 => (1.5, -2.0)        # RR
    )
    
    # Draw root node
    root_circle = plt.Circle((0, 0), 0.12, color="white", ec="black", linewidth=2, zorder=4)
    ax1.add_patch(root_circle)
    ax1.text(0, 0, "Root", ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)
    
    # Determine chosen path for highlighting
    chosen_path_idx = findfirst(p -> (tree.paths[p,1]==res.actions[1] && tree.paths[p,2]==res.actions[2]), 1:4)
    chosen_nodes = tree.path_nodes[chosen_path_idx]
    
    # Draw tree edges
    # Root to L and R
    if 1 in chosen_nodes
        ax1.plot([0, -1], [0, -1], "g-", linewidth=3, alpha=0.7, label="Chosen path")
    else
        ax1.plot([0, -1], [0, -1], "k-", linewidth=2, label="Left")
    end
    if 2 in chosen_nodes
        ax1.plot([0, 1], [0, -1], "g-", linewidth=3, alpha=0.7)
    else
        ax1.plot([0, 1], [0, -1], "k-", linewidth=2, label="Right")
    end
    # L to LL and LR
    if 3 in chosen_nodes
        ax1.plot([-1, -1.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([-1, -1.5], [-1, -2], "k-", linewidth=1.5)
    end
    if 4 in chosen_nodes
        ax1.plot([-1, -0.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([-1, -0.5], [-1, -2], "k-", linewidth=1.5)
    end
    # R to RL and RR
    if 5 in chosen_nodes
        ax1.plot([1, 0.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([1, 0.5], [-1, -2], "k-", linewidth=1.5)
    end
    if 6 in chosen_nodes
        ax1.plot([1, 1.5], [-1, -2], "g-", linewidth=2.5, alpha=0.7)
    else
        ax1.plot([1, 1.5], [-1, -2], "k-", linewidth=1.5)
    end
    
    # Draw nodes with rewards
    for (node_idx, (x, y)) in coords
        if node_idx > 0
            # Highlight chosen leaf
            is_chosen_leaf = (node_idx in chosen_nodes && node_idx >= 3)
            node_color = is_chosen_leaf ? "lightgreen" : "lightblue"
            node_ec = is_chosen_leaf ? "green" : "black"
            node_lw = is_chosen_leaf ? 3 : 2
            
            # Node circle
            circle = plt.Circle((x, y), 0.15, color=node_color, ec=node_ec, linewidth=node_lw, zorder=3)
            ax1.add_patch(circle)
            
            # Node label
            ax1.text(x, y + 0.35, node_names[node_idx], ha="center", fontsize=12, fontweight="bold")
            
            # Reward value
            r_val = res.R_true[node_idx]
            ax1.text(x, y - 0.35, "R=$(round(r_val, digits=2))", ha="center", fontsize=10, 
                    color="red", fontweight="bold")
        end
    end
    
    # Mark observed nodes in stage 1
    obs_stage1_set = Set(res.observed_nodes_stage1)
    for node_idx in obs_stage1_set
        if node_idx in keys(coords)
            x, y = coords[node_idx]
            circle = plt.Circle((x, y), 0.2, color="yellow", alpha=0.5, ec="orange", 
                              linewidth=3, zorder=2)
            ax1.add_patch(circle)
        end
    end
    
    # Mark observed nodes in stage 2
    obs_stage2_set = Set(res.observed_nodes_stage2)
    for node_idx in obs_stage2_set
        if node_idx in keys(coords)
            x, y = coords[node_idx]
            circle = plt.Circle((x, y), 0.2, color="lightgreen", alpha=0.5, ec="green", 
                              linewidth=3, zorder=2)
            ax1.add_patch(circle)
        end
    end
    
    # Add policy posterior text below tree for each observation
    y_text_start = -2.6
    y_spacing = 0.15
    
    # Stage 1 observations
    for (obs_idx, node_idx) in enumerate(res.observed_nodes_stage1)
        if obs_idx <= length(res.policy_posteriors_stage1)
            Π = res.policy_posteriors_stage1[obs_idx+1]  # +1 because first is initial
            posteriors_str = "S1-Obs$obs_idx (node $(node_names[node_idx])): "
            for (p_idx, p_name) in enumerate(path_names)
                posteriors_str *= "$p_name=$(round(Π[p_idx], digits=2)) "
            end
            ax1.text(0, y_text_start - (obs_idx-1)*y_spacing, posteriors_str, 
                    ha="center", fontsize=8, bbox=Dict("facecolor"=>"yellow", "alpha"=>0.3, "boxstyle"=>"round"))
        end
    end
    
    # Stage 2 observations
    stage1_count = length(res.observed_nodes_stage1)
    for (obs_idx, node_idx) in enumerate(res.observed_nodes_stage2)
        if obs_idx <= length(res.policy_posteriors_stage2)
            Π = res.policy_posteriors_stage2[obs_idx+1]  # +1 because first is initial
            posteriors_str = "S2-Obs$obs_idx (node $(node_names[node_idx])): "
            for (p_idx, p_name) in enumerate(path_names)
                if Π[p_idx] > 0.01  # only show non-zero posteriors
                    posteriors_str *= "$p_name=$(round(Π[p_idx], digits=2)) "
                end
            end
            ax1.text(0, y_text_start - (stage1_count + obs_idx - 1)*y_spacing, posteriors_str, 
                    ha="center", fontsize=8, bbox=Dict("facecolor"=>"lightgreen", "alpha"=>0.3, "boxstyle"=>"round"))
        end
    end
    
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-3.5 - (stage1_count + length(res.observed_nodes_stage2))*y_spacing, 0.8)
    ax1.set_title("Decision Tree with Rewards\n(Yellow=Stage1 observed, Green=Stage2 observed)", 
                 fontsize=14, fontweight="bold")
    
    # Policy posterior evolution - Stage 1
    ax2 = subplot(2, 2, 3)
    n_obs_stage1 = length(res.observed_nodes_stage1)
    x_pos = collect(0:n_obs_stage1)
    path_colors = ["blue", "red", "green", "orange"]
    
    for (path_idx, path_name) in enumerate(path_names)
        posteriors = [Π[path_idx] for Π in res.policy_posteriors_stage1]
        ax2.plot(x_pos, posteriors, "o-", label=path_name, alpha=0.8, 
                linewidth=2, markersize=6, color=path_colors[path_idx])
    end
    
    ax2.set_xlabel("Time Step", fontsize=11)
    ax2.set_ylabel("Policy Posterior", fontsize=11)
    ax2.set_title("Stage 1: Policy Posterior Evolution", fontsize=12, fontweight="bold")
    # Show time steps, with intervals if too many
    if length(x_pos) <= 10
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([string(i) for i in x_pos], fontsize=9)
    else
        step = max(1, div(length(x_pos), 10))
        tick_positions = collect(0:step:n_obs_stage1)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([string(i) for i in tick_positions], fontsize=9)
    end
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Policy posterior evolution - Stage 2
    ax3 = subplot(2, 2, 4)
    n_obs_stage2 = length(res.observed_nodes_stage2)
    x_pos2 = collect(0:n_obs_stage2)
    path_colors = ["blue", "red", "green", "orange"]
    
    # Only show feasible paths in stage 2
    feasible_paths = [p for p in 1:4 if any(Π[p] > 0.01 for Π in res.policy_posteriors_stage2)]
    
    for path_idx in feasible_paths
        posteriors = [Π[path_idx] for Π in res.policy_posteriors_stage2]
        ax3.plot(x_pos2, posteriors, "o-", label=path_names[path_idx], alpha=0.8,
                linewidth=2, markersize=6, color=path_colors[path_idx])
    end
    
    ax3.set_xlabel("Time Step", fontsize=11)
    ax3.set_ylabel("Policy Posterior", fontsize=11)
    ax3.set_title("Stage 2: Policy Posterior Evolution", fontsize=12, fontweight="bold")
    # Show time steps, with intervals if too many
    if length(x_pos2) <= 10
        ax3.set_xticks(x_pos2)
        ax3.set_xticklabels([string(i) for i in x_pos2], fontsize=9)
    else
        step = max(1, div(length(x_pos2), 10))
        tick_positions = collect(0:step:n_obs_stage2)
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels([string(i) for i in tick_positions], fontsize=9)
    end
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    PyPlot.savefig("decision_tree_temporal.png", dpi=150, bbox_inches="tight")
    println("\nFigure saved as 'decision_tree_temporal.png'")
    show()
end

# Generate the GIF
create_decision_tree_gif(res, make_tree_2layer_full(), output_path="decision_tree_temporal.gif", frame_duration=0.5, frame_repeat=1)