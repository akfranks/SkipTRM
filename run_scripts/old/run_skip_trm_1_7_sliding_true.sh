export run_name="hidden_512_skips_1_7_mlp_t_sliding_true"

python3 -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1

export TORCH_USE_CUDA_DSA=1

export HYDRA_FULL_ERROR=1

python3 pretrain_c.py \
arch=skip_trm_3 \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True \
+project_name="skip-trm" \
+run_name="${run_name}" \
ema=True
