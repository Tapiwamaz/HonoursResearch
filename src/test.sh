#!/bin/bash

#SBATCH --job-name=Template          # Job name
#SBATCH --output=job_output_%j.log   # Standard output and error log (%j expands to job ID)
#SBATCH --error=job_output_%j.log    # You can separate stdout/stderr if desired
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=4            # CPUs per task
#SBATCH --time=30:30:00              # Time limit hrs:min:sec
#SBATCH --partition=stampede         # Partition name
#SBATCH --mem-per-cpu=2000           # Memory per CPU (in MB, recommended)

# Load required modules (if needed)
# module load python/3.8

# Initialize conda (specific to your cluster)
source /path/to/conda.sh             # Replace with actual conda.sh path

# Activate conda environment
conda activate your_env_name         # Replace with your conda env name

# Go to working directory
cd /HonoursResearch/src      

# Run your Python script
python test.py

# Alternative with more control:
# python -u test.py > output_${SLURM_JOB_ID}.out 2> error_${SLURM_JOB_ID}.err