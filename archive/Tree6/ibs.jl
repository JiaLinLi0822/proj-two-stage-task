using SpecialFunctions: trigamma
using Distributions

mutable struct IBSEstimate{F}
    sample_hit::F
    k::Int
    logp::Float64
end
IBSEstimate(f::Function) = IBSEstimate(f, 1, 0.)

Distributions.var(est::IBSEstimate) = trigamma(1) - trigamma(est.k)
Distributions.mean(est::IBSEstimate) = est.logp


function sample_hit!(est::IBSEstimate)
    if est.sample_hit()
        true
    else
        est.logp -= 1 / (est.k)
        est.k += 1
        false
    end
end

"""
ibs(hit_samplers; repeats, max_iter, neg_logl_threshold)

Arguments:
- hit_samplers: Collection of functions for hit sampling.
- repeats: Number of times to repeat the estimation.
- max_iter: Maximum number of iterations per trial.
- neg_logl_threshold: Threshold for the negative log-likelihood.

Returns:
- neg_logp: The estimated negative log-likelihood.
- std: The standard deviation of the negative log-likelihood estimate.
- converged: Boolean indicating whether the estimation converged.
- n_call: Total number of calls to the hit_samplers.
"""

function ibs(hit_samplers::Vector{<:Function}; repeats=1, max_iter=1000, neg_logp_threshold=Inf)

    total_logp = 0.0
    total_var  = 0.0
    n_call     = 0

    for i in 1:repeats
        # Initialize estimators for each trial
        unconverged     = Set(IBSEstimate(f) for f in hit_samplers)
        converged_logp = 0.0
        converged_var  = 0.0

        while !isempty(unconverged)
            # Accumulate the logp/var of the trials that are not hit yet
            unconverged_logp = 0.0
            unconverged_var  = 0.0
            to_remove        = IBSEstimate[]

            for est in unconverged
                n_call += 1
                if est.k > max_iter || sample_hit!(est)

                    if est.k > max_iter
                        @warn "Termination after maximum number of iterations was reached (the estimate can be arbitrarily biased)."
                    end

                    # Real hit: add to converged
                    converged_logp += mean(est)
                    converged_var  += var(est)
                    push!(to_remove, est)
                else
                    # Continue the trials that are not hit but accumulate their current mean/var
                    unconverged_logp += mean(est)
                    unconverged_var  += var(est)
                end
            end

            # Remove the estimators that are hit
            for est in to_remove
                delete!(unconverged, est)
            end

            # Compute the negative log-likelihood at this moment
            neg_logp = -(converged_logp + unconverged_logp)

            # If the negative log-likelihood exceeds the threshold, treat the trials that are not hit as hits and exit
            if neg_logp > neg_logp_threshold

                @warn "Termination after negative log-likelihood threshold was reached (the estimate can be arbitrarily biased)."

                # Add the trials that are not hit to converged
                converged_logp += unconverged_logp
                converged_var  += unconverged_var

                # The final neg_logp of this repeat
                neg_logp = -converged_logp

                return (
                    neg_logp   = neg_logp,
                    std        = missing,
                    converged  = false,
                    n_call     = n_call
                )
            end
        end

        # If all trials are hit, accumulate the logp/var of this repeat
        total_logp += converged_logp
        total_var  += converged_var
    end

    # Return the average result after all repeats
    avg_logp = total_logp / repeats
    avg_std  = sqrt(total_var / repeats)

    return (
    neg_logp   = -avg_logp,
    std        = avg_std,
    converged  = true,
    n_call     = n_call
    )
end

function ibs(sample_hit::Function, data::Vector; kws...)
    hit_samplers = map(data) do d
        () -> sample_hit(d)
    end
    ibs(hit_samplers; kws...)
end

