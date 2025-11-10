#!/bin/bash
#SBATCH --job-name=DecAE
#SBATCH --output=decode_ae_%j.log
#SBATCH --error=decode_ae_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

# Define input files and output directory
X="../Data/Encoded/cancer-cnn-tanh.npy"
DECODER_PATH="../Models/AE/cnn-decoder.keras"
OUTPUT_DIR="../Results/AE/"
COORDS="../Data/Cancer/cancer-150-1500-h5-coords.npy"
CNN=1

START_TIME=$(date)
echo "Started at: $START_TIME"
python DecAE.py --x "$X" --decoder "$DECODER_PATH" --output "$OUTPUT_DIR" --coords "$COORDS" --cnn "$CNN"

START_TIME=$(date)
echo "Ended at: $START_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
