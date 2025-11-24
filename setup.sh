pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -r requirements.txt
pip install git+https://github.com/lucidrains/adam-atan2-pytorch.git --no-build-isolation
