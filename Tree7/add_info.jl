using Statistics

"""
Calculate stage 1 difficulty: max path reward minus mean of other path rewards.
"""
function calculate_diff1(path::AbstractVector{<:Real})
    idx_max = argmax(path)
    others = [r for (i,r) in enumerate(path) if i != idx_max]
    return path[idx_max] - mean(others)
end


"""
Calculate stage 2 difficulty for Tree2.
If chosen subtree has only one leaf, return -1.
Otherwise, return absolute difference between two leaf values.
"""
function calculate_diff2(v2::AbstractVector{<:Real}, choice1::Int)
    vals = subtree_vals(v2, choice1)
    # If only one leaf in chosen subtree (RL), return -1
    if length(vals) < 2
        return -1.0
    end
    return abs(vals[1] - vals[2])
end

function subtree_vals(v2::AbstractVector{<:Real}, choice1::Int)
    if choice1 == 1
        # Left subtree: indices 1,2
        return v2[1:2]
    elseif choice1 == 2
        # Middle subtree: indices 3,4
        return v2[3:4]
    else
        # Right subtree: index 5
        return [v2[5]]
    end
end

function correct1(best_path_idx::Int, choice1::Int)
    return (best_path_idx ∈ [1,2]) && (choice1 == 1) || 
           (best_path_idx ∈ [3,4]) && (choice1 == 2) || 
           (best_path_idx == 5) && (choice1 == 3)
end

function correct2(v2::AbstractVector{<:Real}, choice1::Int, choice2::Int)
    if choice1 == 1
        # Left subtree: choice2 ∈ {1,2} for leaves 1,2
        vals = subtree_vals(v2, choice1)  # [v2[1], v2[2]]
        return vals[choice2] == maximum(vals)
    elseif choice1 == 2
        # Middle subtree: choice2 ∈ {3,4} for leaves 3,4
        vals = subtree_vals(v2, choice1)  # [v2[3], v2[4]]
        local_choice2 = choice2 - 2  # Convert global index to local index
        return vals[local_choice2] == maximum(vals)
    else
        # Right subtree: choice2 = 5 for leaf 5 (only one option, always correct if reached)
        return true
    end
end

function subtree_relation_code(path::AbstractVector{<:Real})
    idx_desc = sortperm(path; rev=true)
    best, second, third, fourth, worst = idx_desc
    subtree = i -> (i <= 2 ? 0 : (i <= 4 ? 1 : 2))  # 0=left, 1=middle, 2=right

    if best == 5
        return 1  # Best in right subtree (single leaf)
    elseif subtree(best) == subtree(second)
        return 2  # Best and second in same subtree
    elseif subtree(best) == subtree(third)
        return 3  # Best and third in same subtree
    elseif subtree(best) == subtree(fourth)
        return 4  # Best and fourth in same subtree
    elseif subtree(best) == subtree(worst)
        return 5  # Best and worst in same subtree
    end
end
