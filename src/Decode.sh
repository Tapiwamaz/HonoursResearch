#!/bin/bash
#SBATCH --job-name=Decode
#SBATCH --output=decode.log
#SBATCH --error=decode_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=stampede
#SBATCH --time=3-00:00:00      # 3 days max runtime 

# Define input files and output directory
CENTROIDS_PATH=""
DECODER_PATH="/"
OUTPUT_DIR=""

python Decode.py --centroids "$CENTROIDS_PATH" --decoder "$DECODER_PATH" --output_dir "$OUTPUT_DIR"

echo "Tapedza!!! Mwari Ngaakudzwe!"