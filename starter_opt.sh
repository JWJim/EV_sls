#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --partition=open
module load julia
export JULIA_NUM_THREADS=48
julia main.jl
# julia testgrid.jl