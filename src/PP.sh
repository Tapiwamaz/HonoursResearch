#!/bin/bash
#SBATCH --job-name=PP
#SBATCH --output=PP.log
#SBATCH --error=PP_err.log
#SBATCH --nodes=1
#SBATCH --partition=bigbatch
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using 4 cores            
#SBATCH --time=2-00:00:00      # 2 days max runtime 

# Define input files and corresponding job names
INPUT_FILES=(
  "../../mass_spec_data/LPS/09102024_Leandrie_LPS_plasma test/09102024_Leandrie_LPS_plasma test/h5 files/091024_15_LPS_LT_1-1660_SN1p0_centroid.imzml" "../../mass_spec_data/LPS/09102024_Leandrie_LPS_plasma test/09102024_Leandrie_LPS_plasma test/h5 files/091024_15_LPS_LT_1-1660_SN1p0_centroid.imzml" "../../mass_spec_data/LPS/09102024_Leandrie_LPS_plasma test/09102024_Leandrie_LPS_plasma test/h5 files/091024_39_SAL_ST_1-1657_SN1p0_centroid.imzml" "../../mass_spec_data/LPS/09102024_Leandrie_LPS_plasma test/09102024_Leandrie_LPS_plasma test/h5 files/091024_11_SAL_LT_plasma_1-1658_SN1p0_centroid.imzml")
JOB_NAMES=("lps-lt-150-1500" "lps_plasma-150-1500" "sal-st-150-1500" "sal-plasma-150-1500")
OUTPUT_DIRS=("../Data/LPS/" "../Data/LPS/" "../Data/LPS/" "../Data/LPS/")

# Iterate over input files and job names
for i in "${!INPUT_FILES[@]}"; do
    INPUT_FILE="${INPUT_FILES[$i]}"
    JOB_NAME="${JOB_NAMES[$i]}"
	OUTPUT_DIR="${OUTPUT_DIRS[$I]}"
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file not found at $INPUT_FILE"
        continue
    fi
 
    # Check if corresponding .ibd file exists
    IBD_FILE="${INPUT_FILE%.*}.ibd"
    if [ ! -f "$IBD_FILE" ]; then
        echo "Error: Corresponding .ibd file not found at $IBD_FILE"
        continue
    fi

    # Run the Python script
    echo "Starting preprocessing for $JOB_NAME at $(date)"
    echo "Using input file: $INPUT_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo "Job name: $JOB_NAME"
    echo "SLURM Job ID: $SLURM_JOB_ID"

    python Preprocess.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$JOB_NAME"

    echo "Preprocessing for $JOB_NAME completed at $(date)"
    echo "Results saved to $OUTPUT_DIR"
 #   echo "Files created:"
 #   echo "  - ${JOB_NAME}_x.npy (normalized intensity matrix)"
    echo "============================================================================"
done

echo "Tapedza!!! Mwari Ngaakudzwe!"
