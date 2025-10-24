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
Get subtree values based on choice1 for Tree1.
Tree1: left subtree has 1 leaf, right subtree has 2 leaves.
"""
function subtree_vals(path_value::Vector{Float64}, choice1::Int)
    if choice1 == 1
        # left subtree: index 1 only
        return [path_value[1], path_value[2]]
    else
        # right subtree: indices 2 and 3
        return [path_value[3]]
    end
end

"""
Calculate stage 2 difficulty for Tree1.
If chosen subtree has only one leaf, return -1.
Otherwise, return absolute difference between two leaf values.
"""
function calculate_diff2(value2::Vector{Float64}, choice1::Int)
    vals = subtree_vals(value2, choice1)
    if length(vals) < 2
        return -1.0
    end
    return abs(vals[1] - vals[2])
end