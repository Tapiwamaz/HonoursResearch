#!/bin/bash
#SBATCH --job-name=KMEANS
#SBATCH --output=KMEANS_%j.log
#SBATCH --error=KMEANS_%j_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Cancer/cancer_150-1500_x.npy"
OUTPUT_DIR="../Results/kmeans"
COORDS="../Data/Cancer/cancer-150-1500-coords.npy"
ENCODER="../Models/AE/encoder_250.keras"
JOB_NAME="cancer_kmeans_baseline"
K_CLUSTERS=2


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

END_TIME=$(date)
echo "K-Means clustering finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
