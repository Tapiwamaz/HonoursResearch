#!/bin/bash
#SBATCH --job-name=H5
#SBATCH --output=3D_plot.log
#SBATCH --error=3D_plot_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="./3d-cancer_x.npy"
MZS="./3d-cancer_mzs.npy"

OUTPUT_DIR="./"
NAME="cancer"


if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    continue
fi


# Run the Python script
echo "Starting OpenH5.py for $NAME at $(date)"
echo "Using input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"

python Plot.py --X "$INPUT_FILE" --mzs "$MZS" --output "$OUTPUT_DIR" --name "$NAME"

echo "$NAME completed at $(date)"
echo "Results saved to $OUTPUT_DIR"


echo "Tapedza!!! Mwari Ngaakudzwe!"
