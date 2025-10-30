#!/bin/bash

# Body Fat Analysis System - Setup Script
# This script sets up both backend (Flask) and frontend (Next.js)

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Body Fat Analysis System - Setup                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_info "Checking Python installation..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        return 0
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher"
        exit 1
    fi
}

# Check Node.js version
check_node() {
    print_info "Checking Node.js installation..."
    if command_exists node; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION found"
        return 0
    else
        print_error "Node.js not found. Please install Node.js 18 or higher"
        exit 1
    fi
}

# Setup Backend
setup_backend() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Setting up Backend (Flask + Python)"
    echo "════════════════════════════════════════════════════════════"
    
    cd backend || { print_error "backend/ directory not found"; exit 1; }
    
    # Create virtual environment
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    print_success "pip upgraded"
    
    # Install Python dependencies in specific order
    echo ""
    print_info "Installing Python dependencies in optimized order..."
    echo "  (This may take 5-10 minutes)"
    echo ""
    
    echo "Installing PyTorch..."
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch installed"
    
    echo "Installing core packages..."
    pip install numpy==1.24.3 scipy scikit-image scikit-learn
    print_success "Core packages installed"
    
    echo "Installing image processing..."
    pip install opencv-python Pillow
    print_success "Image processing libraries installed"
    
    echo "Installing 3D mesh tools..."
    pip install trimesh pyrender rtree
    print_success "3D mesh tools installed"
    
    echo "Installing PIXIE dependencies..."
    pip install kornia yacs face-alignment loguru
    print_success "PIXIE dependencies installed"
    
    echo "Installing utilities..."
    pip install PyYAML tqdm matplotlib pandas
    print_success "Utilities installed"
    
    echo "Installing Flask and web dependencies..."
    pip install Flask==3.0.0 flask-cors==4.0.0
    print_success "Flask installed"
    
    echo "Installing MediaPipe..."
    pip install mediapipe==0.10.8
    print_success "MediaPipe installed"
    
    echo "Installing SMPL-X..."
    pip install smplx==0.1.28
    print_success "SMPL-X installed"
    
    echo "Attempting chumpy installation..."
    pip install chumpy --no-build-isolation || print_warning "Chumpy installation failed, continuing..."
    
    echo "Installing optional YOLO (for better detection)..."
    pip install ultralytics==8.0.227 || print_warning "YOLO installation failed, continuing..."
    
    echo ""
    print_info "Testing core package imports..."
    python -c "import numpy, torch, trimesh; print('✓ Core packages OK')" || print_error "Import test failed"
    python -c "import cv2, mediapipe; print('✓ CV packages OK')" || print_error "CV import test failed"
    python -c "import flask; print('✓ Flask OK')" || print_error "Flask import test failed"
    
    print_success "All Python dependencies installed successfully"
    
    # Create necessary directories
    print_info "Creating backend directories..."
    mkdir -p uploads
    mkdir -p artifacts/processed_images
    mkdir -p artifacts/3d_mesh
    mkdir -p artifacts/results
    mkdir -p artifacts/image_measurements
    mkdir -p models/smplx
    print_success "Backend directories created"
    
    # Check for SMPL-X models
    if [ ! -f "models/smplx/SMPLX_MALE.npz" ]; then
        print_warning "SMPL-X models not found in models/smplx/"
        print_info "Please download SMPL-X models from: https://smpl-x.is.tue.mpg.de/"
        print_info "Required files: SMPLX_MALE.npz, SMPLX_FEMALE.npz, SMPLX_NEUTRAL.npz"
    else
        print_success "SMPL-X models found"
    fi
    
    # Check for ML model
    if [ ! -f "models/prediction_model.pkl" ]; then
        print_warning "ML model (prediction_model.pkl) not found"
        print_info "System will use fallback formula for body fat calculation"
    else
        print_success "ML model found"
    fi
    
    cd ..
}

# Setup Frontend
setup_frontend() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Setting up Frontend (Next.js)"
    echo "════════════════════════════════════════════════════════════"
    
    if [ ! -d "frontend" ]; then
        print_error "frontend/ directory not found"
        exit 1
    fi
    
    cd frontend
    
    # Install Node dependencies
    print_info "Installing Node.js dependencies (this may take a few minutes)..."
    npm install
    print_success "Node.js dependencies installed"
    
    # Create .env.local if it doesn't exist
    if [ ! -f ".env.local" ]; then
        print_info "Creating .env.local file..."
        cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF
        print_success ".env.local created"
    else
        print_info ".env.local already exists, skipping..."
    fi
    
    cd ..
}

# Create run scripts
create_run_scripts() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Creating run scripts"
    echo "════════════════════════════════════════════════════════════"
    
    # Backend run script
    print_info "Creating backend run script..."
    cat > run-backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
python app.py
EOF
    chmod +x run-backend.sh
    print_success "run-backend.sh created"
    
    # Frontend run script
    print_info "Creating frontend run script..."
    cat > run-frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm run dev
EOF
    chmod +x run-frontend.sh
    print_success "run-frontend.sh created"
    
    # Combined run script
    print_info "Creating combined run script..."
    cat > run-all.sh << 'EOF'
#!/bin/bash
echo "Starting Backend and Frontend..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start backend in background
cd backend
source venv/bin/activate
python app.py &
BACKEND_PID=$!
cd ..

echo "✓ Backend starting on http://localhost:8000"
sleep 3

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✓ Frontend starting on http://localhost:3000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Services running. Press Ctrl+C to stop all services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Trap Ctrl+C and kill both processes
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
EOF
    chmod +x run-all.sh
    print_success "run-all.sh created"
}

# Main setup flow
main() {
    check_python
    check_node
    setup_backend
    setup_frontend
    create_run_scripts
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              Setup Complete! 🎉                           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    print_success "All dependencies installed successfully!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Next Steps:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Add SMPL-X models to backend/models/smplx/"
    echo "   Download from: ${BLUE}https://smpl-x.is.tue.mpg.de/${NC}"
    echo "   Required files:"
    echo "   - SMPLX_MALE.npz"
    echo "   - SMPLX_FEMALE.npz"
    echo "   - SMPLX_NEUTRAL.npz"
    echo ""
    echo "2. (Optional) Add your ML model:"
    echo "   - backend/models/prediction_model.pkl"
    echo ""
    echo "3. Start the application:"
    echo ""
    echo "   ${GREEN}./run-all.sh${NC}     - Start both backend and frontend"
    echo ""
    echo "   Or run separately:"
    echo "   ${GREEN}./run-backend.sh${NC}  - Start Flask backend (port 8000)"
    echo "   ${GREEN}./run-frontend.sh${NC} - Start Next.js frontend (port 3000)"
    echo ""
    echo "4. Open your browser:"
    echo "   ${BLUE}http://localhost:3000${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "For troubleshooting, check README.md"
    echo ""
}

# Run main function
main