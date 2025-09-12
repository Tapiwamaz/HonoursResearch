#!/bin/bash
#SBATCH --job-name=NMF
#SBATCH --output=NMF.log
#SBATCH --error=NMF_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Pretrain/hiv_cancer_150-1500.npy"
OUTPUT_DIR="../Models/NMF"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "NMF training started at: $START_TIME"

JOB_NAME="hiv-cancer"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

python TrainNMF.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME" 

END_TIME=$(date)
echo "Autoencoder training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"