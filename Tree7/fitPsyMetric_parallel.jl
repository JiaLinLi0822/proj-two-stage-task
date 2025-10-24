#!/usr/bin/env julia

using Distributed
using Logging
using LinearAlgebra


const MODELS    = ["model1", "model2", "model3", "model4", "model5", "model6"]
const DATA_FILE = "Tree3/data/Tree3_v3.json"
const OPTIMIZER = :de
const NUM_FUNC_EVALS = 10000

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

ws = workers()[1:length(MODELS)]
tasks = []

for i in eachindex(MODELS)
    task = @spawnat ws[i] _run_model(MODELS[i]; data_file=DATA_FILE, optimizer=OPTIMIZER, NumFuncEvals=NUM_FUNC_EVALS)
    push!(tasks, task)
end

results = fetch.(tasks)

for r in results
    df = r.df
    rss = df.rss[1]; bic = df.bic[1]
    @info "Model $(r.model) done: RSS=$(rss), BIC=$(bic)"
end