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
INPUT_FILES=("../Data/Encoded/cancer-h5-200.npy")
OUTPUT_DIR="../Models/NMF"
MZS_FILE="../Data/Cancer/cancer-150-1500-mzs.npy" 
JOB_NAME="cancer-h5-200" 

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over input files
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    # Extract job name from input file

    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file not found at $INPUT_FILE"
        continue
    fi

    # Check if mzs file exists
    if [ ! -f "$MZS_FILE" ]; then
        echo "Error: MZs file not found at $MZS_FILE"
        exit 1
    fi

    # Run the Python script
    echo "Starting NMF analysis for $JOB_NAME at $(date)"
    echo "Using input file: $INPUT_FILE"
    echo "Using mzs file: $MZS_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo "SLURM Job ID: $SLURM_JOB_ID"

    python NMF.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME" --mzs "$MZS_FILE"

    echo "NMF analysis for $JOB_NAME completed at $(date)"
    echo "Results saved to $OUTPUT_DIR"
    echo "Files created:"
    echo "  - ${JOB_NAME}_original_vs_reconstructed.png (Original vs Reconstructed spectra)"
    echo "  - ${JOB_NAME}_component_loadings_%j.png (Component loadings plot)"
    echo "============================================================================"
done

echo "Tapedza!!! Mwari Ngaakudzwe!"
