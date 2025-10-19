"""
Body Fat Prediction API
Integrates image processing, 3D mesh generation, and ML prediction
"""

import os
import json
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import asdict

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import custom modules
from image_prep.front import BodyPhotoPreprocessor
from image_prep.side import SideViewProcessor
from image_prep.back import BackViewProcessor
from utils.body_measurements import BodyCircumferenceEstimator, BodyMeasurements
from utils.volume_calculation import calculate_body_volume_from_photos

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:5173'])

# Configuration
class Config:
    UPLOAD_FOLDER = './backend/uploads'
    PROCESSED_FOLDER = './backend/artifacts/processed_images'
    MESH_FOLDER = './backend/artifacts/3d_mesh'
    MODEL_PATH = './backend/models/prediction_model.pkl'
    SMPLX_MODEL_DIR = './backend/models'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

config = Config()

# Create necessary directories
for folder in [config.UPLOAD_FOLDER, config.PROCESSED_FOLDER, config.MESH_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Load ML model
try:
    with open(config.MODEL_PATH, 'rb') as f:
        prediction_model = pickle.load(f)
    logger.info("Prediction model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load prediction model: {e}")
    prediction_model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def validate_input(data: Dict) -> Tuple[bool, str]:
    """Validate user input data"""
    required_fields = ['height', 'weight', 'age', 'gender']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    try:
        height = float(data['height'])
        weight = float(data['weight'])
        age = int(data['age'])
        
        if not (120 <= height <= 250):
            return False, "Height must be between 120-250 cm"
        if not (30 <= weight <= 300):
            return False, "Weight must be between 30-300 kg"
        if not (10 <= age <= 100):
            return False, "Age must be between 10-100 years"
        if data['gender'] not in ['male', 'female', 'neutral']:
            return False, "Gender must be 'male', 'female', or 'neutral'"
            
    except ValueError:
        return False, "Invalid data format"
    
    return True, "Valid"

def save_uploaded_file(file, prefix: str) -> Optional[str]:
    """Save uploaded file with unique name"""
    if file and allowed_file(file.filename):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        new_filename = f"{prefix}_{timestamp}.{ext}"
        filepath = os.path.join(config.UPLOAD_FOLDER, new_filename)
        
        # Check file size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        
        if size > config.MAX_FILE_SIZE:
            return None
            
        file.save(filepath)
        return filepath
    return None

def preprocess_images(front_path: str, side_path: str, back_path: str) -> Dict:
    """Preprocess all three images"""
    results = {}
    
    # Process front view
    front_processor = BodyPhotoPreprocessor()
    front_output = os.path.join(config.PROCESSED_FOLDER, 'processed_front.jpg')
    front_result = front_processor.preprocess(front_path, front_output, 'front')
    results['front'] = front_result
    
    # Process side view
    side_processor = SideViewProcessor()
    side_output = os.path.join(config.PROCESSED_FOLDER, 'processed_side.jpg')
    side_result = side_processor.process(side_path, side_output)
    results['side'] = side_result
    
    # Process back view
    back_processor = BackViewProcessor()
    back_output = os.path.join(config.PROCESSED_FOLDER, 'processed_back.jpg')
    back_result = back_processor.process(back_path, back_output)
    results['back'] = back_result
    
    # Check if all preprocessing succeeded
    all_success = all(r['success'] for r in results.values())
    
    if all_success:
        return {
            'success': True,
            'front': front_output,
            'side': side_output,
            'back': back_output,
            'messages': [r['message'] for r in results.values()]
        }
    else:
        failed = [f"{view}: {r['message']}" for view, r in results.items() if not r['success']]
        return {
            'success': False,
            'error': ' | '.join(failed)
        }

def extract_measurements(front_path: str, side_path: str, back_path: str, 
                         height: float) -> Dict:
    """Extract body measurements from processed images"""
    try:
        estimator = BodyCircumferenceEstimator(use_yolo=False)
        measurements = estimator.estimate(front_path, side_path, back_path, height)
        
        # Convert to dict, excluding shoulder_width and calf
        measurements_dict = {
            'neck': measurements.neck,
            'abdomen': measurements.abdomen,
            'hip': measurements.hip,
            'thigh': measurements.thigh,
            'knee': measurements.knee,
            'ankle': measurements.ankle,
            'chest': measurements.shoulder_width * 2.5,  # Estimate chest from shoulder
        }
        
        return {
            'success': True,
            'measurements': measurements_dict,
            'full_measurements': asdict(measurements)
        }
    except Exception as e:
        logger.error(f"Measurement extraction failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def generate_3d_mesh(front_path: str, side_path: str, back_path: str,
                     height: float, weight: float, age: int, gender: str,
                     measurements: Dict) -> Dict:
    """Generate 3D mesh and calculate volume"""
    try:
        result = calculate_body_volume_from_photos(
            front_photo=front_path,
            side_photo=side_path,
            back_photo=back_path,
            height=height,
            weight=weight,
            age=age,
            gender=gender,
            measurements=measurements,
            model_dir=config.SMPLX_MODEL_DIR,
            output_dir=config.MESH_FOLDER,
            export_mesh=True
        )
        
        return {
            'success': True,
            'volume_liters': result['volume_liters'],
            'body_density': result['body_density'],
            'mesh_path': result['mesh_path']
        }
    except Exception as e:
        logger.error(f"3D mesh generation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def predict_body_fat(density: float, age: int, weight: float, height: float,
                     measurements: Dict) -> Dict:
    """Predict body fat percentage using ML model"""
    try:
        if prediction_model is None:
            # Fallback: Use Siri equation if model not available
            body_fat = (495 / density) - 450
            method = "Siri Equation (Fallback)"
        else:
            # Prepare features for ML model
            bmi = weight / ((height / 100) ** 2)
            
            features = {
                'Density': density,
                'Age': age,
                'Weight': weight,
                'Height': height,
                'Neck': measurements.get('neck', 38),
                'Chest': measurements.get('chest', 95),
                'Abdomen': measurements.get('abdomen', 82),
                'Hip': measurements.get('hip', 98),
                'Thigh': measurements.get('thigh', 52),
                'Knee': measurements.get('knee', 36),
                'Ankle': measurements.get('ankle', 23),
                'BMI': bmi,
                'Abdomen_to_Hip': measurements.get('abdomen', 82) / measurements.get('hip', 98),
                'Chest_to_Abdomen': measurements.get('chest', 95) / measurements.get('abdomen', 82),
                'Abdomen_to_Height': measurements.get('abdomen', 82) / height
            }
            
            # Convert to numpy array in correct order
            feature_array = np.array([[
                features['Density'], features['Age'], features['Weight'], features['Height'],
                features['Neck'], features['Chest'], features['Abdomen'], features['Hip'],
                features['Thigh'], features['Knee'], features['Ankle'], features['BMI'],
                features['Abdomen_to_Hip'], features['Chest_to_Abdomen'], features['Abdomen_to_Height']
            ]])
            
            body_fat = prediction_model.predict(feature_array)[0]
            method = "ML Model"
        
        # Calculate body composition
        fat_mass = weight * (body_fat / 100)
        lean_mass = weight - fat_mass
        
        # Determine category
        if body_fat < 6:
            category = "Essential Fat"
        elif body_fat < 14:
            category = "Athletic"
        elif body_fat < 18:
            category = "Fitness"
        elif body_fat < 25:
            category = "Acceptable"
        else:
            category = "Obese"
        
        return {
            'success': True,
            'body_fat_percentage': round(body_fat, 1),
            'fat_mass_kg': round(fat_mass, 1),
            'lean_mass_kg': round(lean_mass, 1),
            'category': category,
            'method': method
        }
    except Exception as e:
        logger.error(f"Body fat prediction failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }
@app.route('/')
def index():
    return "Body Fat Prediction API is running."
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': prediction_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check for files
        if 'front' not in request.files or \
           'side' not in request.files or \
           'back' not in request.files:
            return jsonify({'error': 'Missing image files'}), 400
        
        # Get user data
        user_data = {
            'height': request.form.get('height'),
            'weight': request.form.get('weight'),
            'age': request.form.get('age'),
            'gender': request.form.get('gender', 'neutral')
        }
        
        # Validate input
        is_valid, message = validate_input(user_data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Convert to appropriate types
        height = float(user_data['height'])
        weight = float(user_data['weight'])
        age = int(user_data['age'])
        gender = user_data['gender']
        
        # Save uploaded files
        front_path = save_uploaded_file(request.files['front'], 'front')
        side_path = save_uploaded_file(request.files['side'], 'side')
        back_path = save_uploaded_file(request.files['back'], 'back')
        
        if not all([front_path, side_path, back_path]):
            return jsonify({'error': 'Failed to save uploaded files'}), 400
        
        # Step 1: Preprocess images
        logger.info("Step 1: Preprocessing images...")
        preprocess_result = preprocess_images(front_path, side_path, back_path)
        if not preprocess_result['success']:
            return jsonify({'error': preprocess_result['error']}), 400
        
        processed_front = preprocess_result['front']
        processed_side = preprocess_result['side']
        processed_back = preprocess_result['back']
        
        # Step 2: Extract measurements
        logger.info("Step 2: Extracting body measurements...")
        measurements_result = extract_measurements(
            processed_front, processed_side, processed_back, height
        )
        if not measurements_result['success']:
            return jsonify({'error': measurements_result['error']}), 400
        
        measurements = measurements_result['measurements']
        full_measurements = measurements_result['full_measurements']
        
        # Step 3: Generate 3D mesh and calculate volume
        logger.info("Step 3: Generating 3D mesh...")
        mesh_result = generate_3d_mesh(
            processed_front, processed_side, processed_back,
            height, weight, age, gender, measurements
        )
        if not mesh_result['success']:
            return jsonify({'error': mesh_result['error']}), 400
        
        # Step 4: Predict body fat
        logger.info("Step 4: Predicting body fat...")
        prediction_result = predict_body_fat(
            mesh_result['body_density'], age, weight, height, measurements
        )
        if not prediction_result['success']:
            return jsonify({'error': prediction_result['error']}), 400
        
        # Compile results
        response = {
            'success': True,
            'results': {
                'body_fat_percentage': prediction_result['body_fat_percentage'],
                'fat_mass_kg': prediction_result['fat_mass_kg'],
                'lean_mass_kg': prediction_result['lean_mass_kg'],
                'category': prediction_result['category'],
                'body_density': round(mesh_result['body_density'], 4),
                'volume_liters': round(mesh_result['volume_liters'], 2),
                'measurements': full_measurements,
                'mesh_path': mesh_result['mesh_path'],
                'method': prediction_result['method']
            },
            'processed_images': {
                'front': processed_front,
                'side': processed_side,
                'back': processed_back
            }
        }
        
        logger.info(f"Prediction complete: {prediction_result['body_fat_percentage']}% body fat")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/mesh/<filename>', methods=['GET'])
def get_mesh(filename):
    """Serve 3D mesh file"""
    try:
        filepath = os.path.join(config.MESH_FOLDER, secure_filename(filename))
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='model/obj')
        return jsonify({'error': 'Mesh file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image/<folder>/<filename>', methods=['GET'])
def get_image(folder, filename):
    """Serve processed images"""
    try:
        if folder == 'processed':
            base_folder = config.PROCESSED_FOLDER
        else:
            return jsonify({'error': 'Invalid folder'}), 400
            
        filepath = os.path.join(base_folder, secure_filename(filename))
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)