# sampling.jl - Node sampling strategies

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

