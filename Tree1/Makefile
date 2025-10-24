# Makefile for Tree1 model fitting

.PHONY: help
help:
	@echo "Tree1 Model Fitting (48 cores)"
	@echo "==============================="
	@echo "setup          - Setup Julia environment"
	@echo "check          - Check dependencies"
	@echo "submit         - Sequential fitting (recommended)"
	@echo "submit-parallel- Parallel fitting"
	@echo "status         - Check job status"
	@echo "clean          - Clean log files"
	@echo "results        - Show results"

.PHONY: setup
setup:
	mkdir -p logs
	sbatch setup_slurm_env.slurm

.PHONY: check
check:
	julia check_deps.jl

.PHONY: submit
submit:
	mkdir -p logs
	sbatch submit_optimized.slurm

.PHONY: submit-parallel
submit-parallel:
	mkdir -p logs
	sbatch submit_parallel_models.slurm

.PHONY: status
status:
	squeue -u $$USER

.PHONY: clean
clean:
	rm -f logs/*.out logs/*.err

.PHONY: results
results:
	@ls -lt results/Tree1/*.csv 2>/dev/null | head -5 || echo "No result files"
