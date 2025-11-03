#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --output=cnn_%j.log
#SBATCH --error=cnn_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Pretrain/large_data.npy_part0.npy"
OUTPUT_DIR="../Models/AE/"
PARTITIONS=8
ENCODER_PATH="../Models/AE/cnn-encoder.keras"
DECODER_PATH="../Models/AE/cnn-decoder.keras"


mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Autoencoder training started at: $START_TIME"

JOB_NAME="cnn"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Loop over all partitions
for (( PART_NUM=1; PART_NUM<=PARTITIONS; PART_NUM++ )); do
    echo "Starting partition $PART_NUM of $PARTITIONS at $(date)"
    python CNN.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
        --name "$JOB_NAME" \
        --partitions "$PARTITIONS" \
        --partNum "$PART_NUM" \
        --encoder "$ENCODER_PATH" \
        --decoder "$DECODER_PATH"
    echo "Partition $PART_NUM completed at $(date)"
    echo "============================================================================"
done

END_TIME=$(date)
echo "Autoencoder training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
