#!/bin/bash
#SBATCH --job-name=DA
#SBATCH --output=DA_%j.log
#SBATCH --error=DA_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                 
#SBATCH --partition=stampede 
#SBATCH --time=3-00:00:00     

# Define paths
INPUT_FILE="../../mass_spec_data/LPS/09102024_Leandrie_LPS_plasma test/09102024_Leandrie_LPS_plasma test/h5 files/091024_36_LPS_ST_1-1658_SN1p0_centroid.imzml"
OUTPUT_DIR="../PresentationData"
JOB_NAME="LPS_ST_1-1658"
  # Add mzs file path

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi


# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting DA analysis at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Using mzs file: $MZS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Job name: $JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"

python DA.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --job_type "$JOB_NAME" --job_id "$SLURM_JOB_ID"

echo "Data analysis completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Thank the Lord Almighty!!!"
