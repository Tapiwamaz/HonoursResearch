#!/bin/bash
#SBATCH --job-name=AE
#SBATCH --output=AE_%j.log
#SBATCH --error=AE_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 1 day max runtime 

# Define paths
INPUT_FILE="../Data/HIV/hiv_x.npy"
MZS="../Data/HIV/hiv_mzs.npy"
OUTPUT_DIR="../Results/AE/Hiv"
JOB_NAME="hiv"

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
