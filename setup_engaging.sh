#!/bin/bash

# Load modules
module load miniforge

# Create environment if it doesn't exist
if ! conda env list | grep -q skiptrm; then
    echo "Creating conda environment..."
    conda create -n skiptrm python=3.10 -y
fi

# Activate environment
conda activate skiptrm

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install requirements
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip uninstall adam-atan2
pip install hatchling
pip install git+https://github.com/lucidrains/adam-atan2-pytorch.git --no-build-isolation

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints/skip-trm
mkdir -p data

echo "Setup complete! Don't forget to:"
echo "1. Upload/generate your data to data/sudoku-extreme-1k-aug-1000"
echo "2. Activate the environment with: conda activate skiptrm"

python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

sbatch run_scripts/run_lstm.sh 50000 10000
