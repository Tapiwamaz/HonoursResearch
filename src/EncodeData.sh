#!/bin/bash
#SBATCH --job-name=Encode
#SBATCH --output=Encode.log
#SBATCH --error=Encode_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Cancer/cancer-150-1500-h5-data.npy"
OUTPUT_DIR="../Data/Encoded"
ENCODER_PATH="../Models/Decoder/encoder_h5_wmse_200.keras"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Started at: $START_TIME"

JOB_NAME="cancer-h5-200"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi
if [ ! -f "$ENCODER_PATH" ]; then
    echo "Error: Encoder file not found at $ENCODER_PATH"
    exit 1
fi

python EncodeData.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
        --name "$JOB_NAME" \
        --encoder "$ENCODER_PATH"


END_TIME=$(date)
echo "Data embedding done at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
