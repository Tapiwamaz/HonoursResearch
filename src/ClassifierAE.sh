#!/bin/bash
#SBATCH --job-name=AEBase
#SBATCH --output=AEBase.log
#SBATCH --error=AEBase_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00

INPUT_DATA="../Data/Pretrain/lps-cls1_data.npy"
INPUT_LABELS="../Data/Pretrain/lps-cls1_labels.npy"
AE_ENCODER="../Models/AE/10k-encoder.keras"
OUTPUT_DIR="../Output/AE"
NAME="10k-plasma"

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$INPUT_DATA" ]; then
    echo "Error: Input data file not found at $INPUT_DATA"
    exit 1
fi

if [ ! -f "$INPUT_LABELS" ]; then
    echo "Error: Input labels file not found at $INPUT_LABELS"
    exit 1
fi

if [ ! -f "$AE_ENCODER" ]; then
    echo "Error: Autoencoder encoder file not found at $AE_ENCODER"
    exit 1
fi

START_TIME=$(date)
echo "ClassifierAE training started at: $START_TIME"

python ClassifierAE.py \
    --x "$INPUT_DATA" \
    --y "$INPUT_LABELS" \
    --output "$OUTPUT_DIR" \
    --encoder "$AE_ENCODER" \
    --name "$NAME"

END_TIME=$(date)
echo "ClassifierAE training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
