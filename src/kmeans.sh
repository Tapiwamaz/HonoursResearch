#!/bin/bash
#SBATCH --job-name=KMEANS
#SBATCH --output=kmeans_%j.log
#SBATCH --error=kmeans_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Cancer/cancer_150-1500_x.npy"
OUTPUT_DIR="../Models/Decoder/kmeans"
COORDS="../Data/Cancer/cancer-150-1500-coords.npy"
ENCODER="../Models/Decoder/encoder.keras"
JOB_NAME="cancer"
K_CLUSTERS=3


mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "K-Means clustering started at: $START_TIME"



if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

if [ ! -f "$ENCODER" ]; then
    echo "Error: Encoder file not found at $ENCODER"
    exit 1
fi

if [ ! -f "$COORDS" ]; then
    echo "Error: Coordinates file not found at $COORDS"
    exit 1
fi

python Kmeans.py --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
        --coords "$COORDS" \
        --encoder "$ENCODER" \
        --name "$JOB_NAME" \
        --k "$K_CLUSTERS"

# python Kmeans.py --input "$INPUT_FILE" \
#         --output "$OUTPUT_DIR" \
#         --encoder "$ENCODER" \
#         --name "$JOB_NAME"       

END_TIME=$(date)
echo "K-Means clustering finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
