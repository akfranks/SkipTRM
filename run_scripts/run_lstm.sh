#!/bin/bash
# Remember to run from root
# Run standard lstm

#SBATCH --job-name=lstm_autoresume
#SBATCH --output=logs/lstm_%j.out
#SBATCH --error=logs/lstm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --requeue

# Parse arguments
MAX_EPOCHS=${1:-50000}
EVAL_INTERVAL=${2:-5000}

# Load modules
module load miniforge

# Activate environment
source activate skiptrm

# Set run name
export run_name="standard_run_lstm_autoresume"

# CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Create checkpoint directory
mkdir -p checkpoints/lstm

# Build dataset if it doesn't exist
if [ ! -d "data/sudoku-extreme-1k-aug-1000" ]; then
    echo "Building dataset..."
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 \
        --num-aug 1000
fi

# Function to submit next job
submit_next_job() {
    echo "Submitting continuation job..."
    sbatch "$0" "$MAX_EPOCHS" "$EVAL_INTERVAL"
}

# Trap to resubmit job before timeout
trap submit_next_job SIGUSR1

# Run training
echo "Starting training at $(date)"
echo "Max epochs: $MAX_EPOCHS, Eval interval: $EVAL_INTERVAL"

python3 pretrain.py \
    arch=lstm \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=$MAX_EPOCHS \
    eval_interval=$EVAL_INTERVAL \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    +project_name="organized_runs" \
    +run_name="${run_name}" \
    ema=True

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
else
    echo "Training interrupted at $(date), resubmitting..."
    submit_next_job
fi
