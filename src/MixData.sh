#!/bin/bash
#SBATCH --job-name=MixData
#SBATCH --output=MixData.log
#SBATCH --error=MixData_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00

INPUT_ONE="../Data/Pretrain/hiv_150_1500_h5_x.npy"
INPUT_TWO="../Data/Pretrain/cancer_150_1500_h5_x.npy"
OUTPUT_DIR="../Data/Mixed"
MIXED_NAME="hiv-cancer-150-1500-h5-bigb"
LABELED=0
PARTS=1

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Mixing data started at: $START_TIME"

python MixingData.py \
    --inputOne "$INPUT_ONE" \
	--inputTwo "$INPUT_TWO" \
    --output "$OUTPUT_DIR" \
    --name "$MIXED_NAME" \
    --labeled "$LABELED" \
    --parts "$PARTS"

END_TIME=$(date)
echo "Mixing data finished at: $END_TIME"
echo "Done!"
