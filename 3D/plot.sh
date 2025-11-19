#!/bin/bash
#SBATCH --job-name=H5
#SBATCH --output=3D_plot_%j.log
#SBATCH --error=3D_plot_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../../mass_spec_data/HIV/3 June/3 June PHRU FFPE test 1_1-115501_SN0p0_profile.h5"
#INPUT_FILE="../../mass_spec_data/Cancer biopsy/"
OUTPUT_DIR="./cancer"
NAME="cancer"
COORDS="../Data/Cancer/cancer-150-1500_coords.npy"

mkdir -p "$OUTPUT_DIR"


if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    continue
fi
terter

# Run the Python script
echo "Starting OpenH5.py for $NAME at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"

python Plot.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --name "$NAME" --coords "$COORDS"

echo "$NAME completed at $(date)"
echo "Results saved to $OUTPUT_DIR"


echo "Tapedza!!! Mwari Ngaakudzwe!"
