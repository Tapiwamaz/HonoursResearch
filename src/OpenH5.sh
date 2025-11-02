#!/bin/bash
#SBATCH --job-name=H5
#SBATCH --output=H5.log
#SBATCH --error=H5_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../../mass_spec_data/LPS/09102024_Leandrie_LPS_plasma test/09102024_Leandrie_LPS_plasma test/h5 files/091024_39_SAL_ST.h5"
OUTPUT_DIR="../Data/LPS/"
NAME="sal-st-h5"



mkdir -p "$OUTPUT_DIR"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    continue
fi


# Run the Python script
echo "Starting OpenH5.py for $NAME at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"

python Openh5.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$NAME"

echo "$NAME completed at $(date)"
echo "Results saved to $OUTPUT_DIR"


echo "Tapedza!!! Mwari Ngaakudzwe!"
