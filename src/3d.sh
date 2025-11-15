#!/bin/bash
#SBATCH --job-name=H5
#SBATCH --output=3D.log
#SBATCH --error=3D_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=biggpu
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../../mass_spec_data/Cancer biopsy/5 June/5 June tumour test 2_1-327482_SN0p0_profile.h5"
OUTPUT_DIR="../3D/"
NAME="cancer"



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

python 3D.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$NAME"

echo "$NAME completed at $(date)"
echo "Results saved to $OUTPUT_DIR"


echo "Tapedza!!! Mwari Ngaakudzwe!"
