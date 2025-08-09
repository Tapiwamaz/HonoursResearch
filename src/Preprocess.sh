#!/bin/bash
#SBATCH --job-name=PreprocessHIV
#SBATCH --output=Preporocess.log
#SBATCH --error=PreprocessErr.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # Using 8 cores            
#SBATCH --time=1-00:00:00      # 1 day max runtime (adjust as needed)

# Create log directory
# mkdir -p logs

# Define paths
IMZML_PATH="../../mass_spec_data/HIV/3 June/3 June PHRU FFPE test 1_1-115501_SN0p0_1-56679_SN1p0_centroid.imzml"
# IMZML_PATH="../../mass_spec_data/HIV/3 June/HIV.imzml"

OUTPUT_DIR="./"

# Check if files exist
if [ ! -f "$IMZML_PATH" ]; then
    echo "Error: imzML file not found at $IMZML_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting processing at $(date)"
echo "Using input file: $IMZML_PATH"
echo "Output directory: $OUTPUT_DIR"

python test.py --input "$IMZML_PATH" --output "$OUTPUT_DIR"

echo "Job completed at $(date)"

echo "Job completed successfully"
