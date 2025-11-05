#!/bin/bash
set -e


echo "Installing dependencies with numpy 1.24.3 lock..."

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

pip install --upgrade pip

pip install numpy==1.24.3

pip install -r reqs.txt --no-deps

pip install scipy==1.10.1
pip install scikit-learn==1.3.2
pip install scikit-image==0.21.0
pip install opencv-python==4.8.0.74
pip install Pillow==10.0.0
pip install mediapipe==0.10.8
pip install trimesh==3.23.5
pip install pyrender==0.1.45
pip install rtree==1.0.1
pip install smplx==0.1.13
pip install Flask==2.3.3
pip install flask-cors==4.0.0
pip install Werkzeug==2.3.7
pip install PyYAML==6.0.1
pip install tqdm==4.66.1
pip install matplotlib==3.7.2
pip install pandas==2.0.3
pip install ultralytics==8.0.227 --no-deps || echo "YOLO skipped"
pip install gunicorn==20.1.0

echo "Verifying numpy version..."
python -c "import numpy; assert numpy.__version__ == '1.24.3', f'Wrong numpy: {numpy.__version__}'; print('✓ numpy 1.24.3 locked')"

echo "Testing imports..."
python -c "import numpy, torch, smplx, trimesh, mediapipe, cv2, flask; print('✓ All imports OK')"

echo "Done!"