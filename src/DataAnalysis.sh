#!/bin/bash
#SBATCH --job-name=DA
#SBATCH --output=DA_%j.log
#SBATCH --error=DA_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4      # Using 4 cores            
#SBATCH --time=1-00:00:00      # 1 day max runtime (adjust as needed)

# Define paths
INPUT_FILE="../../mass_spec_data/HIV/3 June/3 June PHRU FFPE test 1_1-115501_SN0p0_profile.h5"
COORDINATES_FILE="../Data/HIV/HIV_Coordinates.npy"
OUTPUT_DIR="../Output"

# Check if input files exist
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

if [ ! -f "$COORDINATES_FILE" ]; then
    echo "Error: Coordinates file not found at $COORDINATES_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting plot generation at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Using coordinates file: $COORDINATES_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"

python DataAnalysis.py --input "$INPUT_FILE" --coordinates "$COORDINATES_FILE" --output "$OUTPUT_DIR" --job_id "$SLURM_JOB_ID"

echo "Plot generation completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Thank the Lord Almighty!!!"
