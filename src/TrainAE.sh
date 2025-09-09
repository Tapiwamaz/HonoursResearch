#!/bin/bash
#SBATCH --job-name=AE
#SBATCH --output=AE.log
#SBATCH --error=AE_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 

INPUT_FILE="../Data/Pretrain/hiv_cancer_150-1500.npy"
OUTPUT_DIR="../Models/AE"
PARTITIONS=20
ENCODER_PATH="$OUTPUT_DIR/ae_encoder_mixed.keras"

echo "tanh activations"
mkdir -p "$OUTPUT_DIR"

START_TIME=$(date)
echo "Autoencoder training started at: $START_TIME"

JOB_NAME=$(basename "$INPUT_FILE" .npy)

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    exit 1
fi

# Loop over all partitions
for (( PART_NUM=2; PART_NUM<=PARTITIONS; PART_NUM++ )); do
    echo "Starting partition $PART_NUM of $PARTITIONS at $(date)"
    python TrainAE.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
        --name "$JOB_NAME" \
        --partitions "$PARTITIONS" \
        --partNum "$PART_NUM" \
        --encoder "$ENCODER_PATH"
    echo "Partition $PART_NUM completed at $(date)"
    echo "============================================================================"
done

END_TIME=$(date)
echo "Autoencoder training finished at: $END_TIME"
echo "Tapedza!!! Mwari Ngaakudzwe!"
