#!/bin/bash
#SBATCH --job-name=tree2_recovery_pda
#SBATCH --output=logs/recovery_pda_%j.out
#SBATCH --error=logs/recovery_pda_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jl16057@nyu.edu

# Parameter recovery for model 6 with PDA (Probability Density Approximation).
# Use (cpus-per-task - 1) as the number of Julia workers; 1 CPU for the main process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs
mkdir -p results/parameter_recovery

NWORKERS=$((${SLURM_CPUS_PER_TASK:-21} - 1))
NWORKERS=$((NWORKERS > 0 ? NWORKERS : 1))

echo "=========================================="
echo "Parameter Recovery: Model 6 (PDA)"
echo "=========================================="
echo "Start time: $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-21}"
echo "Julia workers (parallel subjects): $NWORKERS"
echo "Working directory: $SCRIPT_DIR"
echo "=========================================="

export PATH="${SCRATCH:-$HOME}/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${SCRATCH:-$HOME}/.julia}"

# Avoid oversubscription: one thread per process
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo ""
echo "Julia version:"
julia --version

echo ""
echo "Running PDA parameter recovery (parallel)..."
julia -p "$NWORKERS" --startup-file=no parameter_recovery_model6_pda.jl

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
