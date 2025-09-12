#!/bin/bash
#SBATCH --job-name=PCA
#SBATCH --output=PCA.log
#SBATCH --error=PCA_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Pretrain/hiv_cancer_150-1500.npy"
OUTPUT_DIR="../Models/PCA"
CLASSIFIER_DATA="../Data/Pretrain/sal-lps-150-1500(labeled)_data.npy"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "PCA training started at: $START_TIME"

JOB_NAME="hiv-cancer"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

python TrainPCA.py --input "$INPUT_FILE" \
      --output "$OUTPUT_DIR" \
      --name "$JOB_NAME" \
      --classifier_data "$CLASSIFIER_DATA"

END_TIME=$(date)
echo "Autoencoder training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
