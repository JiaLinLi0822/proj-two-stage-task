# utils.jl - Utility functions

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

