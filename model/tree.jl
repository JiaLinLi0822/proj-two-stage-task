# tree.jl - Decision tree structure

using LinearAlgebra

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
    # 如需认为顶层左右(1,2)也相对"临近"，可解除下一行注释，使 dist(1,2)=1
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

