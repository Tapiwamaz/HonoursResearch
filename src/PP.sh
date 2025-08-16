#!/bin/bash
#SBATCH --job-name=PP
#SBATCH --output=PP_%j.log
#SBATCH --error=PP_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4      # Using 4 cores            
#SBATCH --time=2-00:00:00      # 2 days max runtime 

# Define paths
INPUT_FILE="../../mass_spec_data/EMPA/17092024_cardiac tissues/h5 files/Sample 17_SGLT2_LANME_1-7500_SN1p0_centroid.imzml"
OUTPUT_DIR="../Data/EMPA/"
JOB_NAME="sglt2_lanme"

# Check if input files exist
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Check if corresponding .ibd file exists
IBD_FILE="${INPUT_FILE%.*}.ibd"
if [ ! -f "$IBD_FILE" ]; then
    echo "Error: Corresponding .ibd file not found at $IBD_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting preprocessing at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Job name: $JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"

python Preprocess.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME"

echo "Preprocessing completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Files created:"
echo "  - ${JOB_NAME}_x.npy (normalized intensity matrix)"
echo "  - ${JOB_NAME}_mzs.npy (m/z values)"
echo "  - ${JOB_NAME}_coords.npy (coordinates)"
echo "Thank the Lord Almighty!!!"
