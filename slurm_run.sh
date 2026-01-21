#!/bin/bash
#===============================================================================
# Unified SLURM Wrapper for TinyRecursiveModels
#===============================================================================
# Usage:
#   sbatch slurm_run.sh <experiment_name> [additional_overrides...]
#
# Examples:
#   sbatch slurm_run.sh lstm_base_h512
#   sbatch slurm_run.sh lstm_base_h512 --override epochs=100000
#   sbatch --partition=gpu slurm_run.sh skip_trm_s1248_h256
#
# Environment variables (optional):
#   SLURM_PARTITION  - Override default partition
#   SLURM_TIME       - Override default time limit
#   SLURM_GPUS       - Override default GPU count
#   RUN_SUFFIX       - Add suffix to run name
#===============================================================================

#SBATCH --job-name=trm_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=47:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --requeue

# ==============================================================================
# Configuration
# ==============================================================================

# Get experiment name from argument
EXPERIMENT_NAME="${1:-}"
shift  # Remove first argument, rest are passed to launch.py

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: Experiment name required"
    echo "Usage: sbatch slurm_run.sh <experiment_name> [overrides...]"
    echo "Use 'python launch.py --list' to see available experiments"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Update job name to include experiment
# Note: This doesn't change the job name after submission, but useful for logging
echo "Running experiment: $EXPERIMENT_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# ==============================================================================
# Environment Setup
# ==============================================================================

# Load modules (adjust for your cluster)
module load miniforge 2>/dev/null || true

# Activate environment
source activate skiptrm 2>/dev/null || conda activate skiptrm 2>/dev/null || true

# CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ==============================================================================
# Dataset Check
# ==============================================================================

# Build dataset if it doesn't exist
if [ ! -d "data/sudoku-extreme-1k-aug-1000" ]; then
    echo "Building dataset..."
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 \
        --num-aug 1000
fi

# ==============================================================================
# Auto-Resubmit Setup (for preemptable partitions)
# ==============================================================================

# Function to submit continuation job
submit_next_job() {
    echo "Submitting continuation job..."
    sbatch "$0" "$EXPERIMENT_NAME" "$@"
}

# Trap SIGUSR1 (sent before preemption on some clusters)
trap 'submit_next_job "$@"' SIGUSR1

# ==============================================================================
# Run Training
# ==============================================================================

echo "=============================================="
echo "Starting training: $EXPERIMENT_NAME"
echo "=============================================="

# Build launch command
LAUNCH_ARGS="--experiment $EXPERIMENT_NAME"

# Add run name suffix if specified
if [ -n "$RUN_SUFFIX" ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS --run-name-suffix $RUN_SUFFIX"
fi

# Add any additional arguments
LAUNCH_ARGS="$LAUNCH_ARGS $@"

# Run the launcher
python3 launch.py $LAUNCH_ARGS

# Capture exit code
EXIT_CODE=$?

# ==============================================================================
# Post-Training
# ==============================================================================

echo "=============================================="
echo "Training finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

# Check if training completed successfully or was interrupted
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
elif [ $EXIT_CODE -eq 130 ]; then
    echo "Training interrupted by user (Ctrl+C)"
else
    echo "Training exited with code $EXIT_CODE"
    # Optionally resubmit on failure (for autoresume)
    # Uncomment the following line to enable auto-resubmit on any failure:
    # submit_next_job "$@"
fi

exit $EXIT_CODE
