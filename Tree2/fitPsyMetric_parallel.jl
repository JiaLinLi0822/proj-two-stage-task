#!/usr/bin/env julia

using Distributed
using Logging
using LinearAlgebra


const MODELS    = ["model6", "model7", "model8", "model9", "model10"]
const DATA_FILE = "Tree2/data/Tree2_v3.json"
const OPTIMIZER = :de
const NUM_FUNC_EVALS = 20000 

if nworkers() < length(MODELS)
    addprocs(length(MODELS) - nworkers())
end

Logging.disable_logging(Logging.Warn)

@everywhere begin
    using LinearAlgebra
    LinearAlgebra.BLAS.set_num_threads(1)
    include("fitPsyMetric.jl")
end

@everywhere function _run_model(mname::String; data_file::String, optimizer::Symbol, NumFuncEvals::Int)
    df = main(data_file=data_file, model_name=mname, optimizer=optimizer, NumFuncEvals=NumFuncEvals)
    return (model=mname, df=df)
end

n_workers = min(length(workers()), length(MODELS))
ws = workers()[1:n_workers]
tasks = []

for i in 1:n_workers
    task = @spawnat ws[i] _run_model(MODELS[i]; data_file=DATA_FILE, optimizer=OPTIMIZER, NumFuncEvals=NUM_FUNC_EVALS)
    push!(tasks, task)
end
if n_workers < length(MODELS)
    @warn "Only $n_workers workers available; running first $n_workers models (skipping $(MODELS[n_workers+1:end]))"
end

results = fetch.(tasks)

for r in results
    df = r.df
    rss = df.rss[1]; bic = df.bic[1]
    @info "Model $(r.model) done: RSS=$(rss), BIC=$(bic)"
end