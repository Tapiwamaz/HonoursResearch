#!/bin/bash
#SBATCH --job-name=PretrainAE
#SBATCH --output=PretrainAE_%j.log
#SBATCH --error=PretrainAE_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Pretrain/hiv-cancer-h5-data2.npy"
OUTPUT_DIR="../Models/Decoder/"
PARTITIONS=3
ENCODER_PATH="../Models/Decoder/l-10k.keras"
DECODER_PATH="../Models/Decoder/l-10k_decoder.keras"


mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Autoencoder training started at: $START_TIME"

JOB_NAME=$(basename "$INPUT_FILE" .npy)

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Loop over all partitions
for (( PART_NUM=1; PART_NUM<=PARTITIONS; PART_NUM++ )); do
    echo "Starting partition $PART_NUM of $PARTITIONS at $(date)"
    python PretrainAE.py \
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
