#!/bin/bash
#SBATCH --job-name=tree1_m6
#SBATCH --output=logs/fit_%j.out
#SBATCH --error=logs/fit_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=jl16057@nyu.edu

set -euo pipefail
mkdir -p logs

echo "Start time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"

# module load julia/1.11
export PATH=$SCRATCH/.juliaup/bin:$PATH
export JULIA_DEPOT_PATH=$SCRATCH/.julia

export JULIA_NUM_THREADS=1

srun --cpu-bind=cores --ntasks=1 --cpus-per-task=32 julia -p 31 --startup-file=no submit.jl

echo "End time: $(date)"
