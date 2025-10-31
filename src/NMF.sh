#!/bin/bash
#SBATCH --job-name=NMF
#SBATCH --output=NMF_%j.log
#SBATCH --error=NMF_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu 
#SBATCH --time=3-00:00:00      # 3 days max runtime 

# Define input files and output directory
INPUT_FILES=("../Data/Pretrain/large_data.npy_part0.npy")
OUTPUT_DIR="../Results/NMF/Cancer/"
JOB_NAME="nmf-h5" 
ENCODE="../Data/Cancer/cancer-150-1500-h5-data.npy"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over input files
for INPUT_FILE in "${INPUT_FILES[@]}"; do

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file not found at $INPUT_FILE"
        continue
    fi


    # Run the Python script
    echo "Starting NMF analysis for $JOB_NAME at $(date)"
    echo "Using input file: $INPUT_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo "SLURM Job ID: $SLURM_JOB_ID"

    python NMF.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME" --encode "$ENCODE"

    echo "NMF analysis for $JOB_NAME completed at $(date)"
    echo "Results saved to $OUTPUT_DIR"
    echo "============================================================================"
done

echo "Tapedza!!! Mwari Ngaakudzwe!"
