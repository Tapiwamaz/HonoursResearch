#!/bin/bash
#SBATCH --job-name=DecNMF
#SBATCH --output=decodeNMF.log
#SBATCH --error=decode_nmf_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

# Define input files and output directory
X="../Data/Encoded/nmf-h5_nmf.npy"
DECODER_PATH="../Results/NMF/Cancer/nmf-h5_nmf_model.pkl"
OUTPUT_DIR="../Results/NMF"
COORDS="../Data/Cancer/cancer-150-1500-h5-coords.npy"

START_TIME=$(date)
echo "Started at: $START_TIME"
python DecNMF.py --x "$X" --decoder "$DECODER_PATH" --output "$OUTPUT_DIR" --coords "$COORDS"

START_TIME=$(date)
echo "Ended at: $START_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
