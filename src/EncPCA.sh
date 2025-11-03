#!/bin/bash
#SBATCH --job-name=EncNMF
#SBATCH --output=EncNMF.log
#SBATCH --error=EncNMF_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00

INPUT_DATA="../Data/HIV/hiv-h5_x.npy"
ENCODER="../Results/PCA/Cancer/pca_cancer_pca_model.joblib"
SCALER="../Results/PCA/Cancer/pca_cancer_scaler_model.joblib"
OUTPUT_DIR="../Data/Encoded"
NAME="hiv-pca"

mkdir -p "$OUTPUT_DIR"


if [ ! -f "$INPUT_DATA" ]; then
    echo "Error: Input data file not found at $INPUT_DATA"
    exit 1
fi


START_TIME=$(date)
echo "EncPCA training started at: $START_TIME"


python EncodePCA.py \
    --x "$INPUT_DATA" \
    --output "$OUTPUT_DIR" \
    --encoder "$ENCODER" \
    --name "$NAME" \
    --scaler "$SCALER"

END_TIME=$(date)
echo "EncPCA training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
