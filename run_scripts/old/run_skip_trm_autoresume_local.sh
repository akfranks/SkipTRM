#!/bin/bash

# Local (non-SLURM) version of autoresume training script
# Usage: ./run_skip_trm_autoresume_local.sh [TOTAL_EPOCHS] [EPOCHS_PER_RUN]
# Example: ./run_skip_trm_autoresume_local.sh 50000 10000
#
# This will train for a total of 50000 epochs, running 10000 epochs per job.
# You need to run the script 5 times total to complete all 50000 epochs.
# Each time it will resume from the checkpoint.

# Parse command line arguments
TOTAL_EPOCHS=${1:-50000}
EPOCHS_PER_RUN=${2:-10000}

# Configuration
export run_name="skip_trm_autoresume_${TOTAL_EPOCHS}ep"
export checkpoint_dir="checkpoints/skip-trm/${run_name}"

# CUDA settings
python3 -c "import torch; torch.cuda.empty_cache()"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Print configuration
echo "=========================================="
echo "Local Autoresume Training Configuration"
echo "=========================================="
echo "Total epochs: ${TOTAL_EPOCHS}"
echo "Epochs per run: ${EPOCHS_PER_RUN}"
echo "Run name: ${run_name}"
echo "Checkpoint dir: ${checkpoint_dir}"
echo "=========================================="

# Run training with autoresume support
python3 pretrain_c.py \
arch=skip_trm_3_autoresume \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=${TOTAL_EPOCHS} \
+epochs_per_run=${EPOCHS_PER_RUN} \
eval_interval=5000 \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True \
arch.sliding_skips=true \
+project_name="skip-trm" \
+run_name="${run_name}" \
+checkpoint_path="${checkpoint_dir}" \
ema=True

exit_code=$?

echo "=========================================="
echo "Training finished with exit code: ${exit_code}"
echo "Current epochs completed. To continue training, run this script again."
echo "=========================================="

exit ${exit_code}
