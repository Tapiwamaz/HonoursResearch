#!/bin/bash
#SBATCH --job-name=MixData
#SBATCH --output=MixData.log
#SBATCH --error=MixData_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00

INPUT_ONE="../Data/Pretrain/hiv_cancer_150-1500.npy"
INPUT_TWO="../Data/Cancer/cancer_150-1500_x.npy"
OUTPUT_DIR="../Data/Pretrain"
MIXED_NAME="hiv_cancer_150-1500"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Mixing data started at: $START_TIME"

python MixingData.py \
    --inputOne "$INPUT_ONE" \
    --output "$OUTPUT_DIR" \
    --name "$MIXED_NAME"

END_TIME=$(date)
echo "Mixing data finished at: $END_TIME"
echo "Done!"
