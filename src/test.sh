#!/bin/bash

#SBATCH --job-name=Template          # Job name
#SBATCH --output=job_output_%j.log   # Standard output and error log (%j expands to job ID)
#SBATCH --error=job_output_%j.log    # You can separate stdout/stderr if desired
#SBATCH --ntasks=1                   
#SBATCH --time=02:30:00              # Time limit hrs:min:sec
#SBATCH --partition=stampede         # Partition name          # Memory per CPU (in MB, recommended)
#SBATCH --nodes=1
# Load required modules (if needed)     


python -u test.py > output_${SLURM_JOB_ID}.out 2> error_${SLURM_JOB_ID}.err
echo "Job completed successfully"