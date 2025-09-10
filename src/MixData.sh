#!/bin/bash
#SBATCH --job-name=MixData
#SBATCH --output=MixData.log
#SBATCH --error=MixData_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=biggpu
#SBATCH --time=1-00:00:00

INPUT_ONE="../Data/Cancer/sal-st-150-1500_x.npy"
INPUT_TWO="../Data/Cancer/lps-lt-150-1500_x.npy"
OUTPUT_DIR="../Data/Pretrain"
MIXED_NAME="sal-lps-150-1500(labeled)"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Mixing data started at: $START_TIME"

python MixingData.py \
    --inputOne "$INPUT_ONE" \
	--inputTwo "$INPUT_TWO" \
    --output "$OUTPUT_DIR" \
    --name "$MIXED_NAME"

END_TIME=$(date)
echo "Mixing data finished at: $END_TIME"
echo "Done!"
