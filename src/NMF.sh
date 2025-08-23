#!/bin/bash
#SBATCH --job-name=NMF
#SBATCH --output=NMF_%j.log
#SBATCH --error=NMF_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # Using more cores for biggpu            
#SBATCH --partition=bigbatch 
#SBATCH --time=2-00:00:00      # 1 day max runtime 

# Define paths
INPUT_FILE="../Data/LPS/SAL_LT_plasma_1-1658_x.npy"
OUTPUT_DIR="../Results/NMF/LPS"
JOB_NAME="SAL_LT_plasma_1-1658"
MZS_FILE="../Data/LPS/SAL_LT_plasma_1-1658_mzs.npy"  # Add mzs file path

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Check if mzs file exists
if [ ! -f "$MZS_FILE" ]; then
    echo "Error: MZs file not found at $MZS_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting NMF analysis at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Using mzs file: $MZS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Job name: $JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"

python NMF.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME" --mzs "$MZS_FILE"

echo "NMF analysis completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Files created:"
echo "  - ${JOB_NAME}_original_vs_reconstructed.png (Original vs Reconstructed spectra)"
echo "  - ${JOB_NAME}_component_loadings.png (Component loadings plot)"
echo "Thank the Lord Almighty!!!"
