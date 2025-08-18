#!/bin/bash
#SBATCH --job-name=PP
#SBATCH --output=PP_%j.log
#SBATCH --error=PP_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4      # Using 4 cores            
#SBATCH --time=2-00:00:00      # 2 days max runtime 

# Define paths
INPUT_FILE="../../mass_spec_data/LPS/09102024_Leandrie_LPS_plasma test/09102024_Leandrie_LPS_plasma test/h5 files/091024_39_SAL_ST_1-1657_SN1p0_centroid.imzml"
OUTPUT_DIR="../Data/LPS/"
JOB_NAME="sal_st_3d"

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


# Run the Python script
echo "Starting preprocessing at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Job name: $JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"

python Preprocess3D.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME"

echo "Preprocessing completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Files created:"
echo "  - ${JOB_NAME}_x.npy (normalized intensity matrix)"
echo "Thank the Lord Almighty!!!"
