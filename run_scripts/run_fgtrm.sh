# Remember to run from root
# Run trm with full autograd over all cycles

python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

export run_name="fgtrm"

python3 -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1

export TORCH_USE_CUDA_DSA=1

export HYDRA_FULL_ERROR=1

python3 pretrain.py \
arch=fgtrm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True \
+project_name="organized_runs" \
+run_name="${run_name}" \
ema=True
