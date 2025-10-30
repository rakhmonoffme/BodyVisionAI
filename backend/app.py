import os
# CRITICAL: Set threading limits BEFORE importing numpy/torch/sklearn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import pickle
import uuid
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np

from image_prep.front import BodyPhotoPreprocessor
from image_prep.side import SideViewProcessor
from image_prep.back import BackViewProcessor
from utils.body_measurements import BodyCircumferenceEstimator
from utils.volume_calculation import calculate_body_volume_from_photos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, origins=["http://localhost:3000"])

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'uploads'
PROCESSED_DIR = BASE_DIR / 'artifacts' / 'processed_images'
MESH_DIR = BASE_DIR / 'artifacts' / '3d_mesh'
RESULTS_DIR = BASE_DIR / 'artifacts' / 'results'
MODEL_PATH = BASE_DIR / 'models' / 'prediction_model.pkl'
SMPLX_DIR = BASE_DIR / 'models'
MEASUREMENT_IMG_DIR = BASE_DIR / 'artifacts' / 'image_measurements'

for directory in [UPLOAD_DIR, PROCESSED_DIR, MESH_DIR, RESULTS_DIR, MEASUREMENT_IMG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

try:
    with open(MODEL_PATH, 'rb') as f:
        ML_MODEL = pickle.load(f)
    logger.info("✓ ML model loaded")
except Exception as e:
    ML_MODEL = None
    logger.warning(f"⚠ ML model not found: {e}, using fallback formula")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_body_fat(features):
    """Predict body fat with robust error handling to prevent segfaults"""
    # Try ML model with comprehensive error handling
    if ML_MODEL is not None:
        try:
            feature_order = [ 'Age', 'Weight', 'Height', 'Neck', 'Chest',
                           'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'BMI',
                           'Abdomen_to_Hip', 'Chest_to_Abdomen', 'Abdomen_to_Height']
            
            # Create feature array with explicit dtype to prevent type issues
            feature_values = [features.get(f, 0) for f in feature_order]
            X = np.array([feature_values], dtype=np.float64)
            
            # Debug: log features being fed to model
            logger.info(f"ML Model Input Features:")
            for fname, fval in zip(feature_order, feature_values):
                logger.info(f"  {fname}: {fval}")
            
            # Ensure model prediction happens in isolated manner
            prediction = ML_MODEL.predict(X)
            result = float(prediction[0]) 
            
            # Sanity check: clamp to realistic range (3-50%)
            if result < 3 or result > 50:
                logger.warning(f"ML model returned unrealistic value: {result:.1f}%, clamping to range")
                result = max(3, min(50, result))
            
            logger.info(f"ML model prediction: {result:.1f}%")
            return result
            
        except Exception as e:
            logger.warning(f"ML model prediction failed: {e}, using fallback")
            # Fall through to fallback calculation
    
    # Fallback: Siri equation based on density
    try:
        density = features.get('Density', 1.05)
        body_fat = (4.95 / density - 4.5) * 100 
        
        # Age adjustment
        age = features.get('Age', 30)
        if age > 40:
            body_fat += 2
        elif age < 25:
            body_fat -= 2
        
        # Clamp to reasonable range
        body_fat = max(3, min(50, body_fat))
        logger.info(f"Fallback prediction: {body_fat:.1f}%")
        return body_fat
        
    except Exception as e:
        logger.error(f"Even fallback failed: {e}, using default")
        return 17.1  # Safe default


def assess_health(body_fat, gender, age):
    if gender == 'male':
        if age < 40:
            categories = [(14, 'Athletic'), (18, 'Fit'), (25, 'Acceptable'), (100, 'Obese')]
        else:
            categories = [(16, 'Athletic'), (20, 'Fit'), (28, 'Acceptable'), (100, 'Obese')]
    else:
        if age < 40:
            categories = [(21, 'Athletic'), (25, 'Fit'), (32, 'Acceptable'), (100, 'Obese')]
        else:
            categories = [(23, 'Athletic'), (27, 'Fit'), (35, 'Acceptable'), (100, 'Obese')]
    
    category = 'Obese'
    for threshold, label in categories:
        if body_fat <= threshold:
            category = label
            break
    
    if category in ['Athletic', 'Fit']:
        risk = 'Low'
        recommendation = 'Maintain current healthy body composition'
    elif category == 'Acceptable':
        risk = 'Moderate'
        recommendation = 'Consider increasing physical activity'
    else:
        risk = 'High'
        recommendation = 'Consult healthcare provider for guidance'
    
    return {'category': category, 'risk_level': risk, 'recommendation': recommendation}


def save_session_result(session_id, data):
    result_file = RESULTS_DIR / f'{session_id}.json'
    import json
    with open(result_file, 'w') as f:
        json.dump(data, f)


def load_session_result(session_id):
    result_file = RESULTS_DIR / f'{session_id}.json'
    if result_file.exists():
        import json
        with open(result_file, 'r') as f:
            return json.load(f)
    return None


def process_analysis_sync(session_id, front_path, side_path, back_path, height, weight, age, gender):
    """Synchronous processing - runs in main thread, blocks until complete"""
    try:
        import torch
        torch.set_num_threads(1)
        
        logger.info(f"🔄 Session {session_id} - Starting synchronous processing")
        
        save_session_result(session_id, {
            'status': 'processing',
            'progress': 10,
            'message': 'Validating images...'
        })
        
        front_processor = BodyPhotoPreprocessor()
        side_processor = SideViewProcessor()
        back_processor = BackViewProcessor()
        
        front_processed = str(PROCESSED_DIR / f'{session_id}_front.jpg')
        side_processed = str(PROCESSED_DIR / f'{session_id}_side.jpg')
        back_processed = str(PROCESSED_DIR / f'{session_id}_back.jpg')
        
        logger.info(f"📸 Processing front view...")
        front_result = front_processor.preprocess(front_path, front_processed, 'front')
        if not front_result['success']:
            error_data = {'status': 'error', 'message': f"Front: {front_result['message']}"}
            save_session_result(session_id, error_data)
            return error_data
        
        save_session_result(session_id, {'status': 'processing', 'progress': 20, 'message': 'Processing side view...'})
        
        logger.info(f"📸 Processing side view...")
        side_result = side_processor.process(side_path, side_processed)
        if not side_result['success']:
            error_data = {'status': 'error', 'message': f"Side: {side_result['message']}"}
            save_session_result(session_id, error_data)
            return error_data
        
        save_session_result(session_id, {'status': 'processing', 'progress': 30, 'message': 'Processing back view...'})
        
        logger.info(f"📸 Processing back view...")
        back_result = back_processor.process(back_path, back_processed)
        if not back_result['success']:
            error_data = {'status': 'error', 'message': f"Back: {back_result['message']}"}
            save_session_result(session_id, error_data)
            return error_data
        
        save_session_result(session_id, {'status': 'processing', 'progress': 40, 'message': 'Extracting measurements...'})
        
        logger.info(f"📏 Extracting measurements...")
        estimator = BodyCircumferenceEstimator(use_yolo=False)
        measurements = estimator.estimate(
            front_processed, side_processed, back_processed, height,
            output_dir=str(MEASUREMENT_IMG_DIR), 
            session_id=session_id
        )
        
        logger.info(f"✓ MediaPipe measurements extracted")
        logger.info(f"   Neck: {measurements.neck}cm, Knee: {measurements.knee}cm, Ankle: {measurements.ankle}cm")
        
        save_session_result(session_id, {'status': 'processing', 'progress': 60, 'message': 'Generating 3D model...'})
        
        logger.info(f"🎨 Generating 3D model...")
        
        # Pass initial measurements to 3D model for fitting
        volume_measurements = {
            'neck': measurements.neck,
            'shoulder_width': measurements.shoulder_width,
            'abdomen': measurements.abdomen,
            'hip': measurements.hip,
            'thigh': measurements.thigh,
            'knee': measurements.knee,
            'calf': measurements.calf
        }
        
        volume_result = calculate_body_volume_from_photos(
            front_photo=front_processed,
            side_photo=side_processed,
            back_photo=back_processed,
            height=height,
            weight=weight,
            age=age,
            gender=gender,
            measurements=volume_measurements,
            model_dir=str(SMPLX_DIR),
            output_dir=str(MESH_DIR),
            export_mesh=True,
            session_id=session_id 
        )
        
        # Get refined measurements from 3D model (more accurate than 2D estimates)
        mesh_measurements = volume_result.get('mesh_measurements', {})
        
        # Build final measurements combining 3D mesh (chest, abdomen, hip, thigh) 
        # with MediaPipe measurements (neck, knee, ankle)
        measurements_dict = {
            # From MediaPipe (body_measurements.py)
            'neck': measurements.neck,
            'knee': measurements.knee,
            'ankle': measurements.ankle,
            
            # From 3D mesh (volume_calculation.py) - more accurate
            'chest': mesh_measurements.get('chest', 95.0),
            'abdomen': mesh_measurements.get('waist', measurements.abdomen),  # mesh uses 'waist' key
            'hip': mesh_measurements.get('hip', measurements.hip),
            'thigh': mesh_measurements.get('thigh', measurements.thigh),
            
            # Additional for compatibility/reference
            'waist': mesh_measurements.get('waist', measurements.abdomen),
            'shoulder_width': measurements.shoulder_width,
            'calf': measurements.calf
        }
        
        logger.info(f"✓ Final measurements - Chest: {measurements_dict['chest']:.1f}cm (3D mesh), "
                   f"Abdomen: {measurements_dict['abdomen']:.1f}cm (3D mesh), "
                   f"Hip: {measurements_dict['hip']:.1f}cm (3D mesh), "
                   f"Thigh: {measurements_dict['thigh']:.1f}cm (3D mesh)")
        
        # Handle mesh file - now saved with correct name from the start
        mesh_filename = None
        if volume_result.get('mesh_path'):
            mesh_path = Path(volume_result['mesh_path'])
            if mesh_path.exists():
                mesh_filename = mesh_path.name
                logger.info(f"✓ Mesh saved: {mesh_filename}")
            else:
                logger.warning(f"âš ï¸  Mesh file not found: {mesh_path}")
        
        save_session_result(session_id, {'status': 'processing', 'progress': 85, 'message': 'Calculating body composition...'})
        
        logger.info(f"🧮 Calculating body composition...")
        
        # Calculate BMI safely
        bmi = weight / ((height / 100) ** 2)
        
        # Build features dictionary with safe defaults
        features = {
            'Age': float(age),
            'Weight': float(weight),
            'Height': float(height),
            'Neck': float(measurements_dict['neck']),
            'Chest': float(measurements_dict['chest']),
            'Abdomen': float(measurements_dict['abdomen']),
            'Hip': float(measurements_dict['hip']),
            'Thigh': float(measurements_dict['thigh']),
            'Knee': float(measurements_dict['knee']),
            'Ankle': float(measurements_dict['ankle']),
            'BMI': float(bmi),
            'Abdomen_to_Hip': float(measurements_dict['abdomen'] / max(measurements_dict['hip'], 1)),
            'Chest_to_Abdomen': float(measurements_dict['chest'] / max(measurements_dict['abdomen'], 1)),
            'Abdomen_to_Height': float(measurements_dict['abdomen'] / height)
        }
        
        # This is where the segfault was occurring - now protected
        body_fat_percentage = predict_body_fat(features)
        fat_mass = weight * (body_fat_percentage / 100)
        lean_mass = weight - fat_mass
        
        health_status = assess_health(body_fat_percentage, gender, age)
        
        result_data = {
            'status': 'completed',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'measurements': measurements_dict,
            'body_composition': {
                'body_fat_percentage': round(body_fat_percentage, 1),
                'fat_mass_kg': round(fat_mass, 1),
                'lean_mass_kg': round(lean_mass, 1),
                'muscle_mass_kg': round(lean_mass * 0.45, 1),
                'bmi': round(bmi, 1),
                'volume_liters': round(volume_result.get('volume_liters', 0), 2),
                'body_density': round(volume_result.get('body_density', 1.05), 4)
            },
            'health_metrics': {
                'waist_to_hip_ratio': round(measurements_dict['waist'] / measurements_dict['hip'], 2),
                'health_status': health_status
            },
            'images': {
                'front': f'{session_id}_front.jpg',
                'side': f'{session_id}_side.jpg',
                'back': f'{session_id}_back.jpg',
                'mesh': mesh_filename
            }
        }
        
        save_session_result(session_id, result_data)
        logger.info(f"✅ Session {session_id} completed - BF: {body_fat_percentage:.1f}%")
        
        return result_data
        
    except Exception as e:
        logger.error(f"❌ Session {session_id} error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        error_data = {
            'status': 'error',
            'message': str(e)
        }
        save_session_result(session_id, error_data)
        return error_data


@app.route('/')
def index():
    return jsonify({'message': 'Body Fat Prediction API', 'status': 'online'})


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': ML_MODEL is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/analyze', methods=['POST'])
def create_analysis():
    """SYNCHRONOUS processing - request waits until complete (30-60 seconds)"""
    try:
        if 'front_photo' not in request.files or 'side_photo' not in request.files or 'back_photo' not in request.files:
            return jsonify({'error': 'All 3 photos required'}), 400
        
        front_file = request.files['front_photo']
        side_file = request.files['side_photo']
        back_file = request.files['back_photo']
        
        for file in [front_file, side_file, back_file]:
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file: {file.filename}'}), 400
        
        height = float(request.form.get('height', 0))
        weight = float(request.form.get('weight', 0))
        age = int(request.form.get('age', 0))
        gender = request.form.get('gender', 'neutral').lower()
        
        if not (100 <= height <= 250):
            return jsonify({'error': 'Height must be 100-250 cm'}), 400
        if not (30 <= weight <= 300):
            return jsonify({'error': 'Weight must be 30-300 kg'}), 400
        if not (18 <= age <= 100):
            return jsonify({'error': 'Age must be 18-100'}), 400
        if gender not in ['male', 'female', 'neutral']:
            return jsonify({'error': 'Invalid gender'}), 400
        
        session_id = str(uuid.uuid4())
        
        front_path = str(UPLOAD_DIR / f'{session_id}_front.jpg')
        side_path = str(UPLOAD_DIR / f'{session_id}_side.jpg')
        back_path = str(UPLOAD_DIR / f'{session_id}_back.jpg')
        
        front_file.save(front_path)
        side_file.save(side_path)
        back_file.save(back_path)
        
        logger.info(f"🎬 Starting synchronous analysis for session {session_id}")
        logger.info(f"   Height: {height}cm, Weight: {weight}kg, Age: {age}, Gender: {gender}")
        
        # Process SYNCHRONOUSLY - request blocks here
        result = process_analysis_sync(
            session_id, front_path, side_path, back_path,
            height, weight, age, gender
        )
        
        # Return completed result immediately
        if result['status'] == 'completed':
            return jsonify({
                'session_id': session_id,
                'status': 'completed',
                **result
            }), 200
        else:
            return jsonify({
                'session_id': session_id,
                **result
            }), 400
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/results/<session_id>')
def get_results(session_id):
    result = load_session_result(session_id)
    if not result:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(result), 200


@app.route('/images/<session_id>/<view>')
def get_image(session_id, view):
    if view not in ['front', 'side', 'back']:
        return jsonify({'error': 'Invalid view'}), 400
    
    filepath = PROCESSED_DIR / f'{session_id}_{view}.jpg'
    if not filepath.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    return send_file(filepath, mimetype='image/jpeg')


@app.route('/mesh/<session_id>')
def get_mesh(session_id):
    filepath = MESH_DIR / f'{session_id}_mesh.obj'
    if not filepath.exists():
        return jsonify({'error': 'Mesh not found'}), 404
    
    return send_file(filepath, mimetype='model/obj')
@app.route('/measurement-images/<session_id>/<view>')
def get_measurement_image(session_id, view):
    if view not in ['front', 'side', 'back']:
        return jsonify({'error': 'Invalid view'}), 400
    
    filepath = MEASUREMENT_IMG_DIR / f'{session_id}_{view}_measurements.jpg'
    if not filepath.exists():
        return jsonify({'error': 'Measurement image not found'}), 404
    
    return send_file(filepath, mimetype='image/jpeg')


if __name__ == '__main__':
    logger.info("🚀 Starting Body Fat Prediction API (SYNCHRONOUS MODE)")
    logger.info(f"📁 Uploads: {UPLOAD_DIR}")
    logger.info(f"🤖 Model: {'Loaded' if ML_MODEL is not None else 'Fallback'}")
    logger.info("⚠️  Note: Requests will block 30-60 seconds during processing")
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)