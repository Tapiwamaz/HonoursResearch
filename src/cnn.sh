#!/bin/bash
#SBATCH --job-name=CNN
#SBATCH --output=cnn_%j.log
#SBATCH --error=cnn_%j_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00

INPUT_FILE="../Data/Pretrain/hiv-cancer-h5-data1.npy"
OUTPUT_DIR="../Models/CNN/"
PARTITIONS=3
NAME="cnn-sigmoid"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Encoder saving started at: $START_TIME"


if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Save the encoder for the first partition only
PART_NUM=1
echo "Saving encoder for partition $PART_NUM of $PARTITIONS at $(date)"
python cnn.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --name "$NAME" \
    --partitions "$PARTITIONS" \
    --partNum "$PART_NUM"
echo "Encoder for partition $PART_NUM saved at $(date)"
echo "============================================================================"

END_TIME=$(date)
echo "Encoder saving finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
