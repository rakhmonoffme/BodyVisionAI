#!/bin/bash
set -e

source venv/bin/activate

echo "Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

echo "Installing core packages..."
pip install numpy==1.24.3 scipy scikit-image scikit-learn

echo "Installing image processing..."
pip install opencv-python Pillow

echo "Installing 3D mesh tools..."
pip install trimesh pyrender rtree

echo "Installing PIXIE dependencies..."
pip install kornia yacs face-alignment loguru

echo "Installing utilities..."
pip install PyYAML tqdm matplotlib pandas

echo "Attempting chumpy installation..."
pip install chumpy --no-build-isolation || echo "Chumpy failed, continuing..."

echo "Done! Testing imports..."
python -c "import numpy, torch, trimesh; print('✓ Core packages OK')"