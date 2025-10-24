using Statistics

"""
Calculate stage 1 difficulty: max path reward minus mean of other path rewards.
"""
function calculate_diff1(rewards::Vector{Float64})
    idx_max = argmax(rewards)
    others = [r for (i, r) in enumerate(rewards) if i != idx_max]
    return rewards[idx_max] - mean(others)
end

"""
Get subtree values based on choice1 for Tree2.
Tree2: both left and right subtrees have 2 leaves each.
"""
function subtree_vals(path_value::Vector{Float64}, choice1::Int)
    if choice1 == 1
        # left subtree: indices 1 and 2
        return path_value[1:2]
    else
        # right subtree: indices 3 and 4
        return path_value[3:4]
    end
end

"""
Calculate stage 2 difficulty: absolute difference between two leaf values in chosen subtree.
"""
function calculate_diff2(value2::Vector{Float64}, choice1::Int)
    vals = subtree_vals(value2, choice1)
    if length(vals) < 2
        return -1.0
    end
    return abs(vals[1] - vals[2])
end