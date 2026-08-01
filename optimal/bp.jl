#= Minimal Julia: tree BP with linear-Gaussian nodes, local evidence accumulation.
   - Each node i has prior N(μ0[i], v0[i]).
   - Observations y ~ N(R_i, σ²) accumulate as natural parameters.
   - For a decision between two subtrees, we compute E[max leaf] by MC (small K).
=#

struct Node
    id::Int
    parent::Int    # -1 for root
    children::Vector{Int}
    mu0::Float64
    v0::Float64
    tau_post::Float64   # precision (1/var), mutable state
    eta_post::Float64   # precision*mean, mutable state
end

function make_node(id; parent=-1, children=Int[], mu0=0.0, v0=1.0)
    Node(id, parent, children, mu0, v0, 1/v0, mu0/v0)
end

"Add a Gaussian observation y ~ N(R_i, σ²) to node i."
function add_obs!(nodes::Vector{Node}, i::Int, y::Float64, σ::Float64)
    nodes[i] = Node(nodes[i].id, nodes[i].parent, nodes[i].children,
                    nodes[i].mu0, nodes[i].v0,
                    nodes[i].tau_post + 1/σ^2,
                    nodes[i].eta_post + y/σ^2)
end

"Posterior mean/var for node i."
function post_mv(n::Node)
    μ = n.eta_post / n.tau_post
    v = 1 / n.tau_post
    return μ, v
end

"Collect leaves under node i."
function leaves_under(nodes, i)
    isempty(nodes[i].children) && return [i]
    v = Int[]
    for c in nodes[i].children
        append!(v, leaves_under(nodes, c))
    end
    return v
end

"Monte Carlo estimate E[max leaf] for subtree at i."
function expected_max_leaf(nodes, i; K=2000)
    L = leaves_under(nodes, i)
    μs = Float64[]; vs = Float64[]
    for ℓ in L
        μ,v = post_mv(nodes[ℓ]); push!(μs, μ); push!(vs, v)
    end
    m = 0.0
    for k in 1:K
        mx = -Inf
        for (μ,v) in zip(μs,vs)
            x = randn()*sqrt(v) + μ
            mx = max(mx, x)
        end
        m += mx
    end
    return m / K
end

"At a decision node with children (two subtrees), compute Q(a)=E[max leaf in that subtree]."
function Q_values_at(nodes, i)
    @assert length(nodes[i].children) >= 2
    [expected_max_leaf(nodes, c; K=1000) for c in nodes[i].children]
end

# --- Demo: a depth-2 binary tree (root -> L/R -> leaves) ---
# IDs: 1(root), 2(L),3(R), 4,5 under L; 6,7 under R
nodes = [
    make_node(1; parent=-1, children=[2,3]),
    make_node(2; parent=1, children=[4,5]),
    make_node(3; parent=1, children=[6,7]),
    make_node(4; parent=2, children=[]),
    make_node(5; parent=2, children=[]),
    make_node(6; parent=3, children=[]),
    make_node(7; parent=3, children=[])
]

# priors (can be nonzero/heterogeneous)
for i in 1:length(nodes)
    nodes[i] = make_node(i; parent=nodes[i].parent, children=nodes[i].children, mu0=0.0, v0=2.0)
end

σ = 0.5
# Suppose we "sample" leaf 4 twice and leaf 6 once:
add_obs!(nodes, 4, 1.2, σ)
add_obs!(nodes, 4, 0.6, σ)
add_obs!(nodes, 6, 1.0, σ)

# Decision at root: choose subtree 2 (L) or 3 (R) by expected max-leaf value
Q = Q_values_at(nodes, 1)  # ≈ [E max {4,5}, E max {6,7}]
@show Q, argmax(Q)  # pick the better subtree