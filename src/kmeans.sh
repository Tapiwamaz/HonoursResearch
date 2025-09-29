#!/bin/bash
#SBATCH --job-name=KMEANS
#SBATCH --output=kmeans.log
#SBATCH --error=kmeans_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Pretrain/cancer_150_1500_h5_x.npy"
OUTPUT_DIR="../Results/kmeans"
COORDS="../Data/Pretrain/cancer_150_1500_h5_x.npy_coords.npy"
ENCODER="../Models/AE/encoder_250_dropout_wmse.keras"
JOB_NAME="cancer_wmse_h5"
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

# python Kmeans.py --input "$INPUT_FILE" \
#         --output "$OUTPUT_DIR" \
#         --encoder "$ENCODER" \
#         --name "$JOB_NAME"       

END_TIME=$(date)
echo "K-Means clustering finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
