#!/bin/bash
#SBATCH --job-name=ClassifierPCA
#SBATCH --output=ClassifierPCA.log
#SBATCH --error=ClassifierPCA_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00

INPUT_DATA="../Models/PCA/pca-classifier-train-data.npy"
INPUT_LABELS="../Data/Pretrain/sal-lps-150-1500(labeled)_labels.npy"
ENCODER="../Results/PCA/Cancer/pca_cancer_pca_model.joblib"
SCALER="../Results/PCA/Cancer/pca_cancer_scaler_model.joblib"
OUTPUT_DIR="../Output/PCA"
NAME="pca"

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$INPUT_DATA" ]; then
    echo "Error: Input data file not found at $INPUT_DATA"
    exit 1
fi

if [ ! -f "$INPUT_LABELS" ]; then
    echo "Error: Input labels file not found at $INPUT_LABELS"
    exit 1
fi

START_TIME=$(date)
echo "ClassifierPCA training started at: $START_TIME"

python ClassifierPCA.py \
    --input_data "$INPUT_DATA" \
    --input_lables "$INPUT_LABELS" \
    --output "$OUTPUT_DIR" \
    --name "$NAME" \
    --encoder "$ENCODER" \
    --scaler "$SCALER"

END_TIME=$(date)
echo "ClassifierPCA training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
