#!/bin/bash
#SBATCH --job-name=PCA
#SBATCH --output=PCA_%j.log
#SBATCH --error=PCA_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=1-00:00:00      # 1 day max runtime 


INPUT_FILES=("../Data/Pretrain/hiv_cancer_150-1500.npy")
OUTPUT_DIR="../Results/PCA/Cancer"
JOB_NAME="pca_cancer" 
ENCODE="../Data/Cancer/cancer-150-1500-h5-data.npy"



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

    # Run the Python script
    echo "Starting PCA analysis for $JOB_NAME at $(date)"
    echo "Using input file: $INPUT_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo "SLURM Job ID: $SLURM_JOB_ID"

    python PCA.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME" --encode "$ENCODE"

    echo "PCA analysis for $JOB_NAME completed at $(date)"
    echo "Results saved to $OUTPUT_DIR"
    
done
echo "Tapedza!!! Mwari Ngaakudzwe!"
