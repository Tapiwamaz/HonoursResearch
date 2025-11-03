#!/bin/bash
#SBATCH --job-name=TSNE
#SBATCH --output=tsne_%j.log
#SBATCH --error=tsne_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Results/PCA/Cancer/pca_cancer_encoded_pca.npy"

OUTPUT_DIR="../tsne"
# COORDS="../Data/HIV/hiv-150-1500_coords.npy"
# ENCODER="../Models/AE/encoder_250_dropout_wmse.keras"
NAME="cancer-pca"
CENTROIDS="../kmeans/pca-cancer_centroids_k2.npy"


mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "TSNE training started at: $START_TIME"



if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# python TSNE.py --input "$INPUT_FILE" \
#         --output "$OUTPUT_DIR" \
#         --encoder "$ENCODER" \
# 		--name "$NAME"\
# 		--centroids "$CENTROIDS"
python TSNE.py --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
		--name "$NAME"\
		--centroids "$CENTROIDS"

END_TIME=$(date)
echo "T-SNE training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
