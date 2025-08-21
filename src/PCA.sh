#!/bin/bash
#SBATCH --job-name=PCA
#SBATCH --output=PCA_%j.log
#SBATCH --error=PCA_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # Using more cores for biggpu            
#SBATCH --partition=biggpu 
#SBATCH --time=1-00:00:00      # 1 day max runtime 

# Define paths
INPUT_FILE="../Data/HIV/hiv_x.npy"
OUTPUT_DIR="../Results/PCA/HIV"
JOB_NAME="hiv"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting PCA analysis at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Job name: $JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"

python PCA.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME"

echo "PCA analysis completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Files created:"
echo "  - PCA_${JOB_NAME}.png (PCA scatter plot)"
echo "Thank the Lord Almighty!!!"
