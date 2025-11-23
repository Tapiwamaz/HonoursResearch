#!/bin/bash
#SBATCH --job-name=H5
#SBATCH --output=3D_plot_%j.log
#SBATCH --error=3D_plot_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../../mass_spec_data/HIV/3 June/3 June PHRU FFPE test 1_1-115501_SN0p0_1-56679_SN1p0_centroid.imzml"
NAME="HIV"
INPUT_FILE="../../mass_spec_data/Cancer biopsy/5 June/5 June tumour test 2_1-327482_SN0p0_1-160000_SN1p0_centroid.imzml"
NAME="Cancer"


if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    continue
fi

# Run the Python script
echo "Starting OpenH5.py for $NAME at $(date)"
echo "Using input file: $INPUT_FILE"

echo "SLURM Job ID: $SLURM_JOB_ID"

python Plot.py --x "$INPUT_FILE"  --name "$NAME"

echo "$NAME completed job at $(date)"


echo "Tapedza!!! Mwari Ngaakudzwe!"
