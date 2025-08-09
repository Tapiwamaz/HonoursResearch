#!/bin/bash
#SBATCH --job-name=PreprocessHIV
#SBATCH --output=Preporocess.log
#SBATCH --error=PreprocessErr.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using 16 cores (2x E5-2680 have 16 cores total)
#SBATCH --mem=32G              # Matches the 32GB system RAM per node
#SBATCH --gres=gpu:2           # Request 2 GPUs per node
#SBATCH --time=1-00:00:00      # 1 day max runtime (adjust as needed)

python -u test.py > output_${SLURM_JOB_ID}.out 2> error_${SLURM_JOB_ID}.err
echo "Job completed successfully"