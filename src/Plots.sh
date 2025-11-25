#!/bin/bash
#SBATCH --job-name=Plots
#SBATCH --output=plots.log
#SBATCH --error=plots_err.log
#SBATCH --nodes=1
#SBATCH --partition=stampede
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:00:30

python AccuracyPlots.py
