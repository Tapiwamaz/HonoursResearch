#!/bin/bash
#SBATCH --job-name=DecPCA
#SBATCH --output=decode_pca.log
#SBATCH --error=decode_pca_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

# Define input files and output directory
X="../Results/PCA/Cancer/pca_cancer_encoded_pca.npy"
DECODER_PATH="../Results/PCA/Cancer/pca_cancer_pca_model.joblib"
OUTPUT_DIR="../Results/PCA"
COORDS="../Data/Cancer/cancer-150-1500-h5-coords.npy"
SCALER="../Results/PCA/Cancer/pca_cancer_pca_model.joblib"

START_TIME=$(date)
echo "Started at: $START_TIME"
python DecPCA.py --x "$X" --decoder "$DECODER_PATH" --output "$OUTPUT_DIR" --coords "$COORDS" --scaler "$SCALER"

START_TIME=$(date)
echo "Ended at: $START_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
