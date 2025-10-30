# Body Fat Analysis System

AI-powered body composition analysis using computer vision and 3D body modeling.

## Features

- 📸 3-view photo analysis (front, side, back)
- 🎯 AI-powered body fat percentage prediction
- 📏 Detailed body measurements extraction
- 🎨 3D body mesh generation (SMPL-X)
- 📊 Comprehensive health metrics
- 🖥️ Modern web interface

---

## Prerequisites

Before running the setup script, ensure you have:

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **Git** (optional, for cloning)

---

## Quick Start

### 1. Clone or Download Project

```bash
git clone <your-repo-url>
cd body-fat-analysis
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Install all Python dependencies
- ✅ Install all Node.js dependencies
- ✅ Create necessary directories
- ✅ Set up environment variables
- ✅ Create run scripts

### 3. Add SMPL-X Models (Required)

Download SMPL-X body models from [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)

Place these files in `backend/models/smplx/`:
- `SMPLX_MALE.npz`
- `SMPLX_FEMALE.npz`
- `SMPLX_NEUTRAL.npz`

### 4. Add ML Model (Optional)

If you have a trained ML model for body fat prediction:
- Place `prediction_model.pkl` in `backend/models/`

If not provided, the system uses a fallback formula based on body density.

### 5. Start the Application

#### Option A: Run Everything Together
```bash
./run-all.sh
```

#### Option B: Run Separately

**Terminal 1 - Backend:**
```bash
./run-backend.sh
```

**Terminal 2 - Frontend:**
```bash
./run-frontend.sh
```

### 6. Open in Browser

Navigate to: **http://localhost:3000**

---

## Project Structure

```
body-fat-analysis/
├── backend/
│   ├── app.py                      # Flask API
│   ├── image_prep/                 # Image preprocessing
│   │   ├── front.py
│   │   ├── side.py
│   │   └── back.py
│   ├── utils/
│   │   ├── body_measurements.py    # Measurement extraction
│   │   └── volume_calculation.py   # 3D mesh generation
│   ├── models/
│   │   ├── prediction_model.pkl    # ML model (optional)
│   │   └── smplx/                  # SMPL-X models (required)
│   ├── artifacts/                  # Generated files
│   │   ├── 3d_mesh/
│   │   ├── processed_images/
│   │   ├── image_measurements/
│   │   └── results/
│   └── uploads/                    # Temporary uploads
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Home page
│   │   ├── analyze/page.tsx        # Upload & input form
│   │   ├── processing/page.tsx     # Processing status
│   │   └── results/[id]/page.tsx   # Results display
│   ├── components/                 # UI components
│   ├── lib/
│   │   └── api.ts                  # Backend API calls
│   ├── store/
│   │   └── app-store.ts            # State management
│   └── types/
│       └── index.ts                # TypeScript types
│
├── setup.sh                        # Setup script
├── run-backend.sh                  # Backend runner
├── run-frontend.sh                 # Frontend runner
├── run-all.sh                      # Combined runner
└── requirements.txt                # Python dependencies
```

---

## API Endpoints

### Backend (Flask) - Port 8000

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/analyze` | POST | Submit photos for analysis |
| `/results/<session_id>` | GET | Get analysis results |
| `/images/<session_id>/<view>` | GET | Get processed images |
| `/measurement-images/<session_id>/<view>` | GET | Get annotated images |
| `/mesh/<session_id>` | GET | Download 3D mesh (.obj) |

---

## Usage

### 1. Upload Photos
- Take or upload 3 photos: front, side, and back view
- Follow photo guidelines for best results

### 2. Enter Information
- Height (cm)
- Weight (kg)
- Age
- Gender

### 3. Wait for Analysis
- Processing takes 30-60 seconds
- Progress bar shows current stage

### 4. View Results
- Body composition metrics
- Detailed measurements
- 3D body model
- Health recommendations

---

## Troubleshooting

### Backend Issues

**Error: "SMPL-X models not found"**
- Download models from https://smpl-x.is.tue.mpg.de/
- Place in `backend/models/smplx/`

**Error: "No module named 'cv2'"**
- Run: `pip install opencv-python --break-system-packages`

**Error: "Port 8000 already in use"**
- Kill existing process: `lsof -ti:8000 | xargs kill -9`
- Or change port in `app.py`

### Frontend Issues

**Error: "Cannot connect to backend"**
- Ensure backend is running on port 8000
- Check `.env.local` has correct URL

**Error: "Module not found"**
- Run: `cd frontend && npm install`

---

## Development

### Backend Development
```bash
cd backend
source venv/bin/activate
python app.py
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Testing API
```bash
curl -X POST http://localhost:8000/analyze \
  -F "front_photo=@front.jpg" \
  -F "side_photo=@side.jpg" \
  -F "back_photo=@back.jpg" \
  -F "height=178" \
  -F "weight=74" \
  -F "age=27" \
  -F "gender=male"
```

---

## Tech Stack

**Backend:**
- Flask (API)
- OpenCV + MediaPipe (Body detection)
- SMPL-X (3D modeling)
- Scikit-learn (ML prediction)

**Frontend:**
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Zustand (State management)
- Three.js (3D visualization)

---

## Requirements

### Minimum Hardware
- 4GB RAM
- 2 CPU cores
- 2GB disk space

### Recommended Hardware
- 8GB+ RAM
- 4+ CPU cores
- GPU (for faster processing)

---

## License

[Your License Here]

## Support

For issues or questions:
- Create an issue on GitHub
- Contact: [your-email@example.com]

---

## Acknowledgments

- SMPL-X body models by MPI
- MediaPipe by Google
- Flask framework
- Next.js framework