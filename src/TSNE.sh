#!/bin/bash
#SBATCH --job-name=TSNE
#SBATCH --output=TSNE_%j.log
#SBATCH --error=TSNE_%j_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Pretrain/cancer_150_1500_h5_x.npy"
OUTPUT_DIR="../Results/tsne"
COORDS="../Data/Pretrain/cancer_150_1500_h5_x.npy_coords.npy"
ENCODER="../Models/AE/encoder_250_dropout_wmse.keras"
NAME="cancer_wmse"
CENTROIDS="../Results/kmeans/cance_h5_wmse_centroids_k3.npy"


mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "TSNE training started at: $START_TIME"



if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

python TSNE.py --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
        --encoder "$ENCODER" \
        --coords "$COORDS" \
		--name "$NAME"

END_TIME=$(date)
echo "T-SNE training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
