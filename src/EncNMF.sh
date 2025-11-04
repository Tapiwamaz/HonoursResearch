#!/bin/bash
#SBATCH --job-name=EncNMF
#SBATCH --output=EncNMF.log
#SBATCH --error=EncNMF_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00

INPUT_DATA="../Data/Cancer/cancer-h5_x.npy"
NMF_ENCODER="../Results/NMF/Cancer/nmf-h5_nmf_model.pkl"
OUTPUT_DIR="../Data/Encoded"
NAME="nmf-h5_nmf"

mkdir -p "$OUTPUT_DIR"


if [ ! -f "$INPUT_DATA" ]; then
    echo "Error: Input data file not found at $INPUT_DATA"
    exit 1
fi

if [ ! -f "$NMF_ENCODER" ]; then
    echo "Error: NMF encoder file not found at $NMF_ENCODER"
    exit 1
fi

START_TIME=$(date)
echo "EncNMF training started at: $START_TIME"


python EncodeNMF.py \
    --x "$INPUT_DATA" \
    --output "$OUTPUT_DIR" \
    --encoder "$NMF_ENCODER" \
    --name "$NAME"

END_TIME=$(date)
echo "EncNMF training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
