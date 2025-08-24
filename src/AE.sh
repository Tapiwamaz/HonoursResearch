#!/bin/bash
#SBATCH --job-name=AE
#SBATCH --output=AE_%j.log
#SBATCH --error=AE_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 1 day max runtime 

# Define paths
INPUT_FILE="../Data/LPS/SAL_LT_plasma_1-1658_x.npy"
MZS="../Data/LPS/SAL_LT_plasma_1-1658_mzs.npy"
OUTPUT_DIR="../Results/AE/LPS"
JOB_NAME="SAL_LT_plasma_1-1658"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

if [ ! -f "$MZS" ]; then
    echo "Error: MZS file not found at $MZS"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting Autoencoder analysis at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Job name: $JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"

python AE.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME" --mzs "$MZS"

echo "Autoencoder analysis completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Thank the Lord Almighty!!!"
