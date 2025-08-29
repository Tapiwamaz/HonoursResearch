#!/bin/bash
#SBATCH --job-name=PCA
#SBATCH --output=PCA_%j.log
#SBATCH --error=PCA_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch 
#SBATCH --time=1-00:00:00      # 1 day max runtime 


INPUT_FILES=("../Data/LPS/LPS_ST_1-1658_x.npy" "../Data/LPS/LPS_LT_1-1660_x.npy" "../Data/LPS/SAL_ST_1-1657_x.npy" "../Data/LPS/SAL_LT_plasma_1-1658_x.npy")
OUTPUT_DIR="../Results/PCA/LPS/"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over input files
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    # Extract job name from input file
    JOB_NAME=$(basename "$INPUT_FILE" .npy)

    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file not found at $INPUT_FILE"
        continue
    fi

    # Run the Python script
    echo "Starting PCA analysis for $JOB_NAME at $(date)"
    echo "Using input file: $INPUT_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo "SLURM Job ID: $SLURM_JOB_ID"

    python PCA.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME"

    echo "PCA analysis for $JOB_NAME completed at $(date)"
    echo "Results saved to $OUTPUT_DIR"
    echo "Files created:"
    echo "  - PCA_${JOB_NAME}.png (PCA scatter plot)"
done
echo "Tapedza!!! Mwari Ngaakudzwe!"
