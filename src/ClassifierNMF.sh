#!/bin/bash
#SBATCH --job-name=ClassifierNMF
#SBATCH --output=ClassifierNMF.log
#SBATCH --error=ClassifierNMF_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00

INPUT_DATA="../Data/Pretrain/sal-lps-150-1500(labeled)_data.npy"
INPUT_LABELS="../Data/Pretrain/sal-lps-150-1500(labeled)_labels.npy"
NMF_ENCODER="../Models/NMF/hiv-cancer_nmf_model.joblib"
OUTPUT_DIR="../Output/NMF"
NAME="nmf-classifier"

mkdir -p "$OUTPUT_DIR"


if [ ! -f "$INPUT_DATA" ]; then
    echo "Error: Input data file not found at $INPUT_DATA"
    exit 1
fi

if [ ! -f "$INPUT_LABELS" ]; then
    echo "Error: Input labels file not found at $INPUT_LABELS"
    exit 1
fi

if [ ! -f "$NMF_ENCODER" ]; then
    echo "Error: NMF encoder file not found at $NMF_ENCODER"
    exit 1
fi

START_TIME=$(date)
echo "ClassifierNMF training started at: $START_TIME"


python ClassifierNMF.py \
    --input_data "$INPUT_DATA" \
    --input_lables "$INPUT_LABELS" \
    --output "$OUTPUT_DIR" \
    --nmf_encoder "$NMF_ENCODER" \
    --name "$NAME"

END_TIME=$(date)
echo "ClassifierNMF training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
