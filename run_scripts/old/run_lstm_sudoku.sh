#!/bin/bash

export run_name="lstm_trm_sudoku_test"

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"

# CUDA debugging settings
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Hydra error reporting
export HYDRA_FULL_ERROR=1

# Run training with LSTM-TRM architecture
python3 pretrain_c.py \
arch=trm_lstm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 \
eval_interval=5000 \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
+project_name="lstm-trm" \
+run_name="${run_name}" \
ema=True
