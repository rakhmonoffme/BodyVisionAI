import numpy as np
import cv2
import mediapipe as mp
import trimesh
import torch
from scipy.optimize import minimize
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

try:
    import smplx
except ImportError:
    raise ImportError("Install: pip install smplx")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MediaPipeLandmarkExtractor:
    
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
    
    def extract_from_image(self, image_path: str) -> Dict[str, np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_world_landmarks:
            raise ValueError(f"No pose detected: {image_path}")
        
        landmarks = {}
        for idx, name in enumerate(self.LANDMARK_NAMES):
            lm = results.pose_world_landmarks.landmark[idx]
            landmarks[name] = np.array([lm.x, lm.y, lm.z])
        
        return landmarks
    
    def extract_from_photos(self, front: str, side: str, back: str) -> Dict[str, np.ndarray]:
        logger.info("Extracting MediaPipe landmarks...")
        
        front_lm = self.extract_from_image(front)
        side_lm = self.extract_from_image(side)
        back_lm = self.extract_from_image(back)
        
        averaged = {}
        for name in self.LANDMARK_NAMES:
            averaged[name] = (front_lm[name] + side_lm[name] + back_lm[name]) / 3.0
        
        return averaged
    
    def calibrate_landmarks(self, landmarks: Dict[str, np.ndarray], height: float) -> Dict[str, np.ndarray]:
        head_y = landmarks['nose'][1]
        ankle_y = (landmarks['left_ankle'][1] + landmarks['right_ankle'][1]) / 2
        mp_height = abs(head_y - ankle_y)
        scale = height / mp_height
        
        calibrated = {name: pos * scale for name, pos in landmarks.items()}
        logger.info(f"Calibrated with scale: {scale:.2f}")
        return calibrated


class SMPLXJointMapper:
    
    JOINT_MAPPING = {
        0: 'pelvis', 1: 'left_hip', 2: 'right_hip', 3: 'spine1',
        4: 'left_knee', 5: 'right_knee', 6: 'spine2', 7: 'left_ankle',
        8: 'right_ankle', 9: 'spine3', 10: 'left_foot', 11: 'right_foot',
        12: 'neck', 13: 'left_collar', 14: 'right_collar', 15: 'head',
        16: 'left_shoulder', 17: 'right_shoulder', 18: 'left_elbow',
        19: 'right_elbow', 20: 'left_wrist', 21: 'right_wrist'
    }
    
    MEDIAPIPE_TO_SMPLX = {
        'left_hip': 'left_hip', 'right_hip': 'right_hip',
        'left_knee': 'left_knee', 'right_knee': 'right_knee',
        'left_ankle': 'left_ankle', 'right_ankle': 'right_ankle',
        'left_shoulder': 'left_shoulder', 'right_shoulder': 'right_shoulder',
        'left_elbow': 'left_elbow', 'right_elbow': 'right_elbow',
        'left_wrist': 'left_wrist', 'right_wrist': 'right_wrist',
        'nose': 'head'
    }
    
    def map_landmarks_to_joints(self, landmarks: Dict[str, np.ndarray]) -> Dict[int, np.ndarray]:
        targets = {}
        
        for mp_name, smplx_name in self.MEDIAPIPE_TO_SMPLX.items():
            for joint_id, joint_name in self.JOINT_MAPPING.items():
                if joint_name == smplx_name:
                    targets[joint_id] = landmarks[mp_name]
        
        targets[0] = (landmarks['left_hip'] + landmarks['right_hip']) / 2
        shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
        targets[12] = shoulder_center + np.array([0, 0.05, 0])
        
        return targets


class SMPLXFitter:
    
    BELLY_VERTICES = [3502, 3505, 3508, 3511, 3514, 3517, 3520, 3523, 3526, 3529]
    
    def __init__(self, model_dir: str = './backend/models'):
        self.model_dir = Path(model_dir)
        self.models = {}
    
    def load_model(self, gender: str) -> smplx.SMPLX:
        if gender in self.models:
            return self.models[gender]
        
        model_file = self.model_dir / 'smplx' / f"SMPLX_{gender.upper()}.npz"
        if not model_file.exists():
            raise FileNotFoundError(f"SMPL-X not found: {model_file}")
        
        logger.info(f"Loading SMPL-X: {gender}")
        self.models[gender] = smplx.create(
            model_path=str(self.model_dir),
            model_type='smplx',
            gender=gender,
            num_betas=10,
            ext='npz'
        )
        return self.models[gender]
    
    def _ellipse_circumference(self, width_cm: float, depth_cm: float) -> float:
        a, b = width_cm / 2, depth_cm / 2
        if a == 0 or b == 0:
            return 0
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
    
    def _measure_torso_at_height(self, vertices: np.ndarray, y_pos: float, 
                                 tolerance: float = 0.03) -> Tuple[float, float]:
        mask = np.abs(vertices[:, 1] - y_pos) < tolerance
        verts = vertices[mask]
        
        if len(verts) < 10:
            return 0.0, 0.0
        
        center_x = verts[:, 0].mean()
        x_std = verts[:, 0].std()
        torso_mask = np.abs(verts[:, 0] - center_x) < (x_std * 1.8)
        verts = verts[torso_mask]
        
        if len(verts) < 10:
            return 0.0, 0.0
        
        front = verts[verts[:, 2] > 0]
        back = verts[verts[:, 2] <= 0]
        
        front_w = (front[:, 0].max() - front[:, 0].min()) * 100 if len(front) > 5 else 0
        back_w = (back[:, 0].max() - back[:, 0].min()) * 100 if len(back) > 5 else 0
        
        if front_w > 0 and back_w > 0:
            width = (front_w + back_w) / 2
        elif front_w > 0:
            width = front_w
        else:
            width = (verts[:, 0].max() - verts[:, 0].min()) * 100
        
        depth = (verts[:, 2].max() - verts[:, 2].min()) * 100
        
        return width, depth
    
    def _measure_limb_at_height(self, vertices: np.ndarray, y_pos: float, 
                                limb_x: float, radius: float = 0.08,
                                tolerance: float = 0.025) -> Tuple[float, float]:
        mask = np.abs(vertices[:, 1] - y_pos) < tolerance
        verts = vertices[mask]
        
        if len(verts) == 0:
            return 0.0, 0.0
        
        limb_mask = np.abs(verts[:, 0] - limb_x) < radius
        limb_verts = verts[limb_mask]
        
        if len(limb_verts) < 5:
            return 0.0, 0.0
        
        width = (limb_verts[:, 0].max() - limb_verts[:, 0].min()) * 100
        depth = (limb_verts[:, 2].max() - limb_verts[:, 2].min()) * 100
        
        return width, depth
    
    def calculate_mesh_measurements(
        self, 
        vertices: np.ndarray, 
        joints: np.ndarray
    ) -> Dict[str, float]:
        measurements = {}
        
        head = joints[15]
        neck_joint = joints[12]
        left_shoulder = joints[16]
        right_shoulder = joints[17]
        spine2 = joints[6]
        left_hip = joints[1]
        right_hip = joints[2]
        left_knee = joints[4]
        left_ankle = joints[7]
        
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_joint_dist = np.linalg.norm(left_shoulder - right_shoulder) * 100
        
        # SHOULDER WIDTH
        measurements['shoulder_width'] = shoulder_joint_dist + 8.0
        
        # NECK - midpoint between head and neck joint
        neck_y = (head[1] + neck_joint[1]) / 2
        neck_width, neck_depth = self._measure_limb_at_height(
            vertices, neck_y, 0.0, radius=0.045, tolerance=0.015
        )
        
        if neck_width == 0:
            neck_width = shoulder_joint_dist * 0.32
            neck_depth = neck_width * 0.88
        elif neck_depth == 0:
            neck_depth = neck_width * 0.88
        
        measurements['neck'] = self._ellipse_circumference(neck_width, neck_depth)
        
        # CHEST - at spine2 (nipple line), below arms in T-pose
        chest_y = spine2[1]
        chest_width, chest_depth = self._measure_torso_at_height(vertices, chest_y, tolerance=0.03)
        
        if chest_width > 0 and chest_depth > 0:
            measurements['chest'] = self._ellipse_circumference(chest_width, chest_depth)
        else:
            measurements['chest'] = 95.0
        
        # ABDOMEN - 35% from shoulder to hip
        abdomen_y = shoulder_mid_y * 0.35 + left_hip[1] * 0.65
        abdomen_width, abdomen_depth = self._measure_torso_at_height(vertices, abdomen_y, tolerance=0.03)
        
        if abdomen_width > 0 and abdomen_depth > 0:
            measurements['abdomen'] = self._ellipse_circumference(abdomen_width, abdomen_depth)
            measurements['waist'] = measurements['abdomen']
        else:
            measurements['abdomen'] = 80.0
            measurements['waist'] = 80.0
        
        # HIP - search for widest point
        hip_mid_y = (left_hip[1] + right_hip[1]) / 2
        best_circumference = 0
        best_width, best_depth = 0, 0
        
        for i in range(15):
            y = hip_mid_y - 0.10 + (0.12 * i / 14)
            width, depth = self._measure_torso_at_height(vertices, y, tolerance=0.035)
            
            if width > 0 and depth > 0:
                circ = self._ellipse_circumference(width, depth)
                if circ > best_circumference:
                    best_circumference = circ
                    best_width, best_depth = width, depth
        
        if best_width > 0 and best_depth > 0:
            measurements['hip'] = self._ellipse_circumference(best_width, best_depth)
        else:
            measurements['hip'] = 95.0
        
        # THIGH - proximal thigh (20% down from hip, at gluteal fold region)
        thigh_y = left_hip[1] + (left_knee[1] - left_hip[1]) * 0.20
        thigh_width, thigh_depth = self._measure_limb_at_height(
            vertices, thigh_y, left_hip[0], radius=0.085, tolerance=0.025
        )
        
        if thigh_width > 0 and thigh_depth > 0:
            measurements['thigh'] = self._ellipse_circumference(thigh_width, thigh_depth)
        else:
            measurements['thigh'] = 52.0
        
        # CALF - 1/3 from knee down
        calf_y = left_knee[1] * 0.67 + left_ankle[1] * 0.33
        calf_width, calf_depth = self._measure_limb_at_height(
            vertices, calf_y, left_knee[0], radius=0.08, tolerance=0.025
        )
        
        if calf_width > 0 and calf_depth > 0:
            measurements['calf'] = self._ellipse_circumference(calf_width, calf_depth)
        else:
            measurements['calf'] = 36.0
        
        # KNEE
        knee_y = left_knee[1]
        knee_width, knee_depth = self._measure_limb_at_height(
            vertices, knee_y, left_knee[0], radius=0.08, tolerance=0.03
        )
        
        if knee_width > 0 and knee_depth > 0:
            measurements['knee'] = self._ellipse_circumference(knee_width, knee_depth)
        else:
            measurements['knee'] = 36.0
        
        return measurements
    
    def optimize_shape(
        self,
        model: smplx.SMPLX,
        target_joints: Dict[int, np.ndarray],
        height: float,
        weight: float,
        measurements: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        
        logger.info("Optimizing SMPL-X shape parameters...")
        if measurements:
            logger.info(f"Using {len(measurements)} measurements as targets")
        
        initial_betas = np.zeros(10)
        if weight:
            bmi = weight / ((height / 100) ** 2)
            if bmi < 18.5:
                initial_betas[0] = -1.5
            elif bmi < 22:
                initial_betas[0] = (bmi - 20) * 0.5
            elif bmi < 25:
                initial_betas[0] = (bmi - 22) * 0.8
            else:
                initial_betas[0] = (bmi - 22) / 3.0
            
            if bmi < 25:
                initial_betas[1] = -0.5
        
        target_joints_array = np.array([target_joints[i] for i in sorted(target_joints.keys())])
        target_joint_ids = sorted(target_joints.keys())
        
        target_shoulder = np.linalg.norm(target_joints[16] - target_joints[17])
        target_hip = np.linalg.norm(target_joints[1] - target_joints[2])
        target_torso = np.linalg.norm(target_joints[12] - target_joints[0])
        
        def objective(betas):
            betas_torch = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
            output = model(betas=betas_torch, return_verts=True)
            joints = output.joints.detach().cpu().numpy()[0]
            vertices = output.vertices.detach().cpu().numpy()[0]
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces)
            predicted_volume = mesh.volume * 1000
            expected_volume = weight / 1.05
            volume_error = ((predicted_volume - expected_volume) ** 2) * 80.0
            
            predicted_height = joints[15][1] - joints[7][1]
            height_error = (predicted_height - height) ** 2 * 50
            
            belly_error = 0
            pelvis_z = joints[0][2]
            belly_vertices = vertices[self.BELLY_VERTICES]
            belly_center_z = belly_vertices[:, 2].mean()
            belly_protrusion = abs(belly_center_z - pelvis_z)
            if belly_protrusion > 0.08:
                belly_error = (belly_protrusion - 0.08) ** 2 * 100.0
            
            measurement_error = 0
            predicted_chest = None
            predicted_waist = None
            
            if measurements:
                mesh_measurements = self.calculate_mesh_measurements(vertices, joints)
                
                for name in measurements.keys():
                    # Skip chest - measured at different body location (spine2 vs shoulder)
                    if name == 'chest':
                        predicted_chest = mesh_measurements.get('chest', 0)
                        continue
                    
                    if name == 'shoulder_width':
                        if 'shoulder_width' in mesh_measurements:
                            predicted = mesh_measurements['shoulder_width']
                            target = measurements['shoulder_width']
                            measurement_error += (predicted - target) ** 2 * 35.0
                        continue
                    
                    mesh_key = name if name != 'abdomen' else 'waist'
                    
                    if mesh_key in mesh_measurements:
                        predicted = mesh_measurements[mesh_key]
                        target = measurements[name]
                        
                        if name in ['waist', 'abdomen']:
                            predicted_waist = predicted
                        
                        if name in ['waist', 'abdomen']:
                            measurement_error += (predicted - target) ** 2 * 50.0
                        elif name == 'hip':
                            measurement_error += (predicted - target) ** 2 * 40.0
                        elif name in ['thigh', 'calf']:
                            measurement_error += (predicted - target) ** 2 * 25.0
                        else:
                            measurement_error += (predicted - target) ** 2 * 20.0
            
            # Chest-to-waist ratio removed since chest not used in optimization
            ratio_error = 0
            
            shoulder_error = (np.linalg.norm(joints[16] - joints[17]) - target_shoulder) ** 2 * 5
            hip_error = (np.linalg.norm(joints[1] - joints[2]) - target_hip) ** 2 * 5
            torso_error = (np.linalg.norm(joints[12] - joints[0]) - target_torso) ** 2 * 5
            
            joint_error = 0
            for i, jid in enumerate(target_joint_ids):
                joint_error += np.sum((joints[jid][:2] - target_joints_array[i][:2]) ** 2) * 0.05
            
            reg = 0
            for i, beta in enumerate(betas):
                if i == 1:
                    reg += beta ** 2 * 0.005
                else:
                    reg += beta ** 2 * 0.01
            
            total = (volume_error + height_error + belly_error + measurement_error + 
                    ratio_error + shoulder_error + hip_error + torso_error + 
                    joint_error + reg)
            
            return total
        
        result = minimize(
            objective, 
            initial_betas, 
            method='L-BFGS-B', 
            bounds=[(-5, 5)] * 10,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        logger.info(f"Optimization: {result.nit} iterations, error: {result.fun:.2f}")
        
        betas_torch = torch.tensor(result.x.reshape(1, -1), dtype=torch.float32)
        output = model(betas=betas_torch, return_verts=True)
        vertices = output.vertices.detach().cpu().numpy()[0]
        joints = output.joints.detach().cpu().numpy()[0]
        
        final_volume = trimesh.Trimesh(vertices=vertices, faces=model.faces).volume * 1000
        logger.info(f"Volume: {final_volume:.2f}L (target: {weight/1.05:.2f}L)")
        
        pelvis_z = joints[0][2]
        belly_vertices = vertices[self.BELLY_VERTICES]
        belly_center_z = belly_vertices[:, 2].mean()
        belly_protrusion = abs(belly_center_z - pelvis_z) * 100
        logger.info(f"Belly: {belly_protrusion:.1f}cm (target: <8cm)")
        
        if measurements:
            mesh_measurements = self.calculate_mesh_measurements(vertices, joints)
            
            logger.info("\n" + "="*60)
            logger.info("MEASUREMENT ACCURACY:")
            logger.info("="*60)
            
            for name in measurements.keys():
                mesh_key = name if name != 'abdomen' else 'waist'
                
                if mesh_key in mesh_measurements:
                    pred = mesh_measurements[mesh_key]
                    target = measurements[name]
                    error = abs(pred - target)
                    error_pct = (error / target * 100) if target > 0 else 0
                    
                    # Mark chest as reference only (not used in optimization)
                    note = " (reference)" if name == 'chest' else ""
                    
                    logger.info(f"  {name:15s}: {pred:5.1f}cm vs {target:5.1f}cm "
                              f"(±{error:4.1f}cm, {error_pct:4.1f}%){note}")
            
            logger.info("="*60)
        
        return result.x
    
    def generate_mesh(self, model: smplx.SMPLX, betas: np.ndarray) -> trimesh.Trimesh:
        betas_torch = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
        output = model(betas=betas_torch, return_verts=True)
        vertices = output.vertices.detach().cpu().numpy()[0]
        mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces)
        
        if not mesh.is_watertight:
            logger.warning("Repairing mesh...")
            mesh.merge_vertices()
            mesh.remove_degenerate_faces()
            mesh.fill_holes()
        
        logger.info(f"Mesh: {len(vertices)} vertices, watertight={mesh.is_watertight}")
        return mesh


def calculate_body_volume_from_photos(
    front_photo: str,
    side_photo: str,
    back_photo: str,
    height: float,
    weight: float,
    age: int,
    gender: str,
    measurements: Optional[Dict[str, float]] = None,
    model_dir: str = './backend/models',
    output_dir: str = './backend/artifacts/3d_mesh',
    export_mesh: bool = True,
    session_id: str = None
) -> Dict:
    """
    Calculate body volume and density from 3 photos using SMPL-X model.
    """
    if gender not in ['male', 'female', 'neutral']:
        raise ValueError(f"Invalid gender: {gender}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info(f"3D Body Volume Calculation")
    logger.info(f"Gender: {gender}, Height: {height}cm, Weight: {weight}kg")
    if measurements:
        logger.info(f"Measurements: {list(measurements.keys())}")
    logger.info("="*60)
    
    extractor = MediaPipeLandmarkExtractor()
    landmarks = extractor.extract_from_photos(front_photo, side_photo, back_photo)
    calibrated = extractor.calibrate_landmarks(landmarks, height)
    
    mapper = SMPLXJointMapper()
    targets = mapper.map_landmarks_to_joints(calibrated)
    
    fitter = SMPLXFitter(model_dir)
    model = fitter.load_model(gender)
    betas = fitter.optimize_shape(model, targets, height, weight, measurements)
    mesh = fitter.generate_mesh(model, betas)
    
    volume_liters = mesh.volume * 1000
    density = weight / volume_liters
    
    logger.info(f"\nRESULTS:")
    logger.info(f"  Volume: {volume_liters:.2f}L")
    logger.info(f"  Density: {density:.4f}kg/L")
    
    if density < 0.95 or density > 1.15:
        logger.warning(f"  ⚠️  Density outside normal range (0.95-1.15)")
    else:
        logger.info(f"  ✓ Density within normal range")
    
    # Calculate final mesh measurements from optimized model
    betas_torch = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
    output = model(betas=betas_torch, return_verts=True)
    vertices = output.vertices.detach().cpu().numpy()[0]
    joints = output.joints.detach().cpu().numpy()[0]
    mesh_measurements = fitter.calculate_mesh_measurements(vertices, joints)
    
    logger.info(f"\n✓ Final measurements: Chest={mesh_measurements.get('chest', 0):.1f}cm, "
               f"Abdomen={mesh_measurements.get('abdomen', 0):.1f}cm, "
               f"Thigh={mesh_measurements.get('thigh', 0):.1f}cm")
    
    mesh_path = None
    if export_mesh:
        # Use session_id if provided, otherwise fallback to gender-based name
        if session_id:
            mesh_filename = f"{session_id}_mesh.obj"
        else:
            mesh_filename = f"body_mesh_{gender}.obj"
        
        mesh_path = str(Path(output_dir) / mesh_filename)
        mesh.export(mesh_path)
        logger.info(f"\nâœ… Mesh: {mesh_path}")
    
    return {
        'volume_liters': volume_liters,
        'body_density': density,
        'is_watertight': mesh.is_watertight,
        'mesh_path': mesh_path,
        'beta_parameters': betas.tolist(),
        'mesh_measurements': mesh_measurements,
        'method': 'SMPLX_CleanMeasurements'
    }


if __name__ == '__main__':
    # Example usage
    measurements = {
        'abdomen': 74,
        'hip': 87,
        'thigh': 48,
        'neck': 40,
        'calf': 36.0,
        'shoulder_width': 44.0,
        'chest': 108.6,  # Reference only, not used in optimization
    }
    
    result = calculate_body_volume_from_photos(
        front_photo='front_view.png',
        side_photo='side_view.png',
        back_photo='back_view.png',
        height=178.0,
        weight=74.0,
        age=27,
        gender='male',
        measurements=measurements,
        model_dir='./backend/models'
    )
    
    print(f"\n{'='*60}")
    print(f"Volume:  {result['volume_liters']:.2f}L")
    print(f"Density: {result['body_density']:.4f}kg/L")
    print(f"Mesh:    {result['mesh_path']}")
    print(f"{'='*60}")