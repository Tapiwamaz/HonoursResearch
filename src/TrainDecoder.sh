#!/bin/bash
#SBATCH --job-name=TrainDecoder
#SBATCH --output=TrainDecoder_%j.log
#SBATCH --error=TrainDecoder_%j_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00

INPUT_FILE="../Data/Encoded/hiv-cancer-encoded.npy"
OUTPUT_DIR="../Models/Decoder"
NAME="decoder_250_dropout_wmse"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Decoder training started at: $START_TIME"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Train the decoder
echo "Training decoder at $(date)"
python TrainDecoder.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --name "$NAME"
echo "Decoder training completed at $(date)"
echo "============================================================================"

END_TIME=$(date)
echo "Decoder training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
