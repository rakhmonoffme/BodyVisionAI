import cv2
import mediapipe as mp
import numpy as np
from typing import Union, Dict, List

class BodyPhotoPreprocessor:
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        self.critical_landmarks = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'shoulders': [11, 12],
            'hips': [23, 24],
            'knees': [25, 26],
            'ankles': [27, 28],
            'feet': [29, 30, 31, 32]
        }
    def preprocess(self, image: Union[str, np.ndarray], 
                   output_path: str = "processed_image.jpg",
                   view_type: str = 'front') -> Dict:
        """Main preprocessing pipeline - full validation for front view only"""
        try:
            img = self._load_image(image)
            if img is None:
                return {'success': False, 'message': 'Error: Could not load image'}
            
            # Full checks only for front view
            if view_type == 'front':
                checks = [
                    self._detect_person,
                    self._check_full_body,
                    self._check_camera_angle,
                    self._check_lighting,
                    self._check_clothing
                ]
            else:
                # Simpler checks for side/back (handled by separate processors)
                checks = [self._detect_person]
            
            landmarks = None
            for check in checks:
                if check == self._detect_person:
                    result = check(img)
                    if not result['success']:
                        return result
                    landmarks = result['landmarks']
                else:
                    result = check(img, landmarks)
                    if not result['success']:
                        return result
            
            # Determine contrasting background and extract body
            bg_color = self._determine_contrast_color(img, landmarks)
            result = self._extract_body(img, landmarks, output_path, bg_color)
            if result['success']:
                result['message'] = f'Image processed successfully (background: {"light" if bg_color > 150 else "dark"})'
            
            return result
            
        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Load image from path or numpy array"""
        if isinstance(image, str):
            return cv2.imread(image)
        return image
    
    def _detect_person(self, image: np.ndarray) -> Dict:
        """Detect exactly one person"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return {'success': False, 'message': 'Error: No person detected. Ensure you are visible in the frame'}
        
        landmarks = results.pose_landmarks.landmark
        visible_count = sum(1 for lm in landmarks if lm.visibility > 0.5)
        
        if visible_count < 20:
            return {'success': False, 'message': 'Error: Multiple people or occlusion detected. Ensure only one person is in frame'}
        
        return {'success': True, 'landmarks': landmarks}
    
    def _check_full_body(self, image: np.ndarray, landmarks: List) -> Dict:
        """Verify full body visibility from head to feet"""
        h, w = image.shape[:2]
        missing = []
        
        for part, indices in self.critical_landmarks.items():
            if not any(landmarks[i].visibility > 0.5 for i in indices):
                missing.append(part)
        
        if missing:
            return {'success': False, 'message': f'Full body not visible. Retake photo. Missing: {", ".join(missing)}'}
        
        # Check only key points (nose for head, ankles for feet) with 10% margin
        key_points = {
            'head': 0,      # Nose
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        margin = 0.10
        for name, idx in key_points.items():
            lm = landmarks[idx]
            if lm.visibility > 0.3:  # Lower threshold for edge check
                x, y = lm.x, lm.y
                if x < margin or x > (1 - margin) or y < margin or y > (1 - margin):
                    return {'success': False, 'message': f'Full body not visible. Retake photo. {name} too close to edge'}
        
        return {'success': True}
    
    def _check_camera_angle(self, image: np.ndarray, landmarks: List) -> Dict:
        """Validate hip-level camera positioning"""
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        # Hips should be in middle third of image
        if hip_y < 0.35 or hip_y > 0.65:
            return {'success': False, 'message': 'Camera angle is not right. Hold camera at hip-level, 2-3 meters away'}
        
        return {'success': True}
    
    def _check_lighting(self, image: np.ndarray, landmarks: List) -> Dict:
        """Validate lighting conditions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        if brightness < 60:
            return {'success': False, 'message': 'Error: Image too dark. Increase lighting'}
        if brightness > 240:
            return {'success': False, 'message': 'Error: Image too bright. Reduce lighting'}
        if contrast < 30:
            return {'success': False, 'message': 'Error: Poor contrast. Ensure even lighting'}
        
        # Check uneven lighting across quadrants
        h, w = gray.shape
        quadrants = [
            gray[0:h//2, 0:w//2], gray[0:h//2, w//2:w],
            gray[h//2:h, 0:w//2], gray[h//2:h, w//2:w]
        ]
        
        if np.std([np.mean(q) for q in quadrants]) > 40:
            return {'success': False, 'message': 'Error: Uneven lighting. Use uniform lighting without shadows'}
        
        return {'success': True}
    
    def _check_clothing(self, image: np.ndarray, landmarks: List) -> Dict:
        """Check for patterned clothing that may affect accuracy"""
        h, w = image.shape[:2]
        
        # Extract torso region
        x_min = int(min(landmarks[11].x, landmarks[23].x) * w) - 20
        x_max = int(max(landmarks[12].x, landmarks[24].x) * w) + 20
        y_min = int(min(landmarks[11].y, landmarks[12].y) * h) - 20
        y_max = int(max(landmarks[23].y, landmarks[24].y) * h) + 20
        
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        torso = image[y_min:y_max, x_min:x_max]
        if torso.size == 0:
            return {'success': True}
        
        # Detect high-contrast patterns
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.15:
            return {'success': False, 'message': 'Warning: Patterned clothing may affect accuracy. Wear plain, form-fitting clothes'}
        
        return {'success': True}
    
    def _determine_contrast_color(self, image: np.ndarray, landmarks: List) -> int:
        """Determine contrasting background color based on person's clothing brightness"""
        h, w = image.shape[:2]
        
        # Sample body area
        points = [[int(lm.x * w), int(lm.y * h)] 
                  for lm in landmarks if lm.visibility > 0.5]
        
        if len(points) < 4:
            return 200
        
        # Create rough body mask
        hull = cv2.convexHull(np.array(points))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Calculate average brightness of person
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        person_pixels = gray[mask > 0]
        
        if len(person_pixels) == 0:
            return 200
        
        avg_brightness = np.mean(person_pixels)
        
        # Dark clothing -> light background, Light clothing -> dark background
        if avg_brightness < 100:
            return 220
        elif avg_brightness > 150:
            return 60
        else:
            return 128
    
    def _extract_body(self, image: np.ndarray, landmarks: List, output_path: str, bg_color: int) -> Dict:
        """Extract person and place on background (255=white, 128=gray for contrast)"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create mask from landmarks
        points = [[int(lm.x * w), int(lm.y * h)] 
                  for lm in landmarks if lm.visibility > 0.5]
        
        if len(points) < 10:
            return {'success': False, 'message': 'Error: Cannot extract body. Ensure good lighting and visibility'}
        
        # Expanded convex hull
        hull = cv2.convexHull(np.array(points))
        center = hull.mean(axis=0)
        expanded_hull = (center + (hull - center) * 1.15).astype(np.int32)
        cv2.fillConvexPoly(mask, expanded_hull, 255)
        
        # Refine with GrabCut
        try:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            mask_gc = np.where(mask == 255, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            
            cv2.grabCut(image, mask_gc, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            final_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except:
            final_mask = mask
        
        # Validate extraction quality
        fg_ratio = np.sum(final_mask > 0) / (h * w)
        if fg_ratio < 0.1 or fg_ratio > 0.8:
            return {'success': False, 'message': 'Error: Background extraction failed. Use plain, contrasting background'}
        
        # Clean and smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
        
        # Blend onto background (white or gray)
        bg = np.ones_like(image) * bg_color
        mask_3ch = np.stack([final_mask / 255.0] * 3, axis=-1)
        result = (image * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
        
        cv2.imwrite(output_path, result)
        return {'success': True, 'output_path': output_path}
    
    def __del__(self):
        self.pose.close()


def main():
    preprocessor = BodyPhotoPreprocessor()
    
    # Process single image
    result = preprocessor.preprocess("front_view.png", "processed_front_view.png")
    print(f"{'✓' if result['success'] else '✗'} {result['message']}")
    

if __name__ == "__main__":
    main()