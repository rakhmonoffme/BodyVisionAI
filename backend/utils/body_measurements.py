import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class BodyMeasurements:
    neck: float
    shoulder_width: float  # NEW: bone-to-bone width
    abdomen: float
    hip: float
    thigh: float
    knee: float
    calf: float  # NEW: calf circumference
    ankle: float

class BodyCircumferenceEstimator:
    def __init__(self, use_yolo: bool = False):
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.use_yolo = use_yolo
        self.yolo_model = self._load_yolo() if use_yolo else None
    
    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            return YOLO('yolov8n.pt')
        except ImportError:
            print("YOLO not available")
            return None
    
    def estimate(self, front_img_path: str, side_img_path: str, 
                 back_img_path: str, height_cm: float) -> BodyMeasurements:
        
        # Read images with error handling
        front_img = cv2.imread(front_img_path)
        side_img = cv2.imread(side_img_path)
        back_img = cv2.imread(back_img_path)
        
        if front_img is None or side_img is None or back_img is None:
            raise ValueError("Could not read one or more image files")
        
        if self.yolo_model:
            front_img = self._crop_person(front_img)
            side_img = self._crop_person(side_img)
            back_img = self._crop_person(back_img)
        
        front_mask = self._get_mask(front_img)
        side_mask = self._get_mask(side_img)
        back_mask = self._get_mask(back_img)
        
        front_lm = self._get_landmarks(front_img)
        side_lm = self._get_landmarks(side_img)
        back_lm = self._get_landmarks(back_img)
        
        front_scale = self._calc_scale(front_mask, height_cm)
        side_scale = self._calc_scale(side_mask, height_cm)
        back_scale = self._calc_scale(back_mask, height_cm)
        
        measurements = self._calculate_circumferences(
            front_mask, side_mask, back_mask,
            front_lm, side_lm, back_lm,
            front_scale, side_scale, back_scale
        )
        
        self._visualize(front_img, side_img, back_img,
                       front_mask, side_mask, back_mask,
                       front_lm, side_lm, back_lm,
                       measurements)
        
        return measurements
    
    def _crop_person(self, img: np.ndarray) -> np.ndarray:
        results = self.yolo_model(img, classes=[0], verbose=False)
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            return img[box[1]:box[3], box[0]:box[2]]
        return img
    
    def _get_mask(self, img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        return mask
    
    def _get_landmarks(self, img: np.ndarray) -> Optional[Dict]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            h, w = img.shape[:2]
            landmarks = {}
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = (int(lm.x * w), int(lm.y * h), lm.visibility)
            return landmarks
        return None
    
    def _calc_scale(self, mask: np.ndarray, height_cm: float) -> float:
        rows_with_person = np.where(np.any(mask > 0, axis=1))[0]
        if len(rows_with_person) == 0:
            return height_cm / mask.shape[0]
        
        top = rows_with_person[0]
        bottom = rows_with_person[-1]
        pixel_height = bottom - top
        
        return height_cm / pixel_height if pixel_height > 0 else height_cm / mask.shape[0]
    
    def _get_contiguous_width(self, mask: np.ndarray, y: int, x_center: int, search_range: int = 5) -> Tuple[int, int, int]:
        """Find contiguous body segment width around x_center at y position"""
        h, w = mask.shape
        y_start = max(0, y - search_range)
        y_end = min(h, y + search_range)
        
        best_width = 0
        best_x1, best_x2 = x_center, x_center
        
        for row in range(y_start, y_end):
            row_mask = mask[row, :]
            
            if row_mask[x_center] > 0:
                x1 = x_center
                while x1 > 0 and row_mask[x1 - 1] > 0:
                    x1 -= 1
                
                x2 = x_center
                while x2 < w - 1 and row_mask[x2 + 1] > 0:
                    x2 += 1
                
                width = x2 - x1
                if width > best_width:
                    best_width = width
                    best_x1, best_x2 = x1, x2
        
        return best_x1, best_x2, best_width
    
    def _measure_torso_width(self, mask: np.ndarray, y: int, torso_center_x: int, search_range: int = 5) -> Tuple[int, int, int]:
        """Measure torso width excluding arms"""
        h, w = mask.shape
        y_start = max(0, y - search_range)
        y_end = min(h, y + search_range)
        
        all_x1, all_x2 = [], []
        
        for row in range(y_start, y_end):
            row_mask = mask[row, :]
            if row_mask[torso_center_x] == 0:
                continue
            
            segments = []
            in_segment = False
            start = 0
            
            for x in range(w):
                if row_mask[x] > 0 and not in_segment:
                    start = x
                    in_segment = True
                elif row_mask[x] == 0 and in_segment:
                    segments.append((start, x - 1))
                    in_segment = False
            
            if in_segment:
                segments.append((start, w - 1))
            
            for x1, x2 in segments:
                if x1 <= torso_center_x <= x2:
                    all_x1.append(x1)
                    all_x2.append(x2)
                    break
        
        if all_x1 and all_x2:
            x1 = int(np.median(all_x1))
            x2 = int(np.median(all_x2))
            return x1, x2, x2 - x1
        
        return torso_center_x, torso_center_x, 0
    
    def _measure_neck(self, front_mask, side_mask, front_lm, side_lm, 
                     front_scale, side_scale) -> Tuple[float, float, float]:
        """
        Neck measurement at base of neck (midpoint between nose and shoulders)
        """
        if not front_lm or 0 not in front_lm or 11 not in front_lm or 12 not in front_lm:
            return 12.0, 11.0, 38.0
        
        nose = front_lm[0]
        left_shoulder = front_lm[11]
        right_shoulder = front_lm[12]
        
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) // 2
        
        neck_x = (nose[0] + shoulder_mid_x) // 2
        neck_y = (nose[1] + shoulder_mid_y) // 2
        
        shoulder_width_px = abs(right_shoulder[0] - left_shoulder[0])
        x1_f, x2_f, w_f = self._get_contiguous_width(front_mask, neck_y, neck_x, search_range=5)
        
        if w_f == 0:
            neck_width_px = shoulder_width_px * 0.40
        else:
            neck_width_px = w_f
        
        neck_width_cm = neck_width_px * front_scale
        
        if side_lm and 0 in side_lm and 11 in side_lm:
            side_nose = side_lm[0]
            side_shoulder = side_lm[11]
            side_neck_y = (side_nose[1] + side_shoulder[1]) // 2
            
            _, _, neck_depth_px = self._get_contiguous_width(side_mask, side_neck_y, 
                                                            side_mask.shape[1] // 2, search_range=5)
            
            if neck_depth_px == 0:
                neck_depth_cm = neck_width_cm * 0.90
            else:
                neck_depth_cm = neck_depth_px * side_scale
        else:
            neck_depth_cm = neck_width_cm * 0.90
        
        a = neck_width_cm / 2
        b = neck_depth_cm / 2
        h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0
        neck_circumference = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        
        return neck_width_cm, neck_depth_cm, neck_circumference
    
    def _measure_shoulder_width(self, front_lm, front_scale) -> float:
        """
        Measure shoulder width between shoulder landmarks
        Add correction factor to approximate acromion-to-acromion distance
        """
        if not front_lm or 11 not in front_lm or 12 not in front_lm:
            return 42.0
        
        left_shoulder = front_lm[11]
        right_shoulder = front_lm[12]
        
        # Distance between shoulder joints
        shoulder_width_px = abs(right_shoulder[0] - left_shoulder[0])
        
        # Add ~8cm to approximate acromion bones (which extend beyond joints)
        shoulder_width_cm = (shoulder_width_px * front_scale) + 8.0
        
        return shoulder_width_cm
    
    def _measure_calf(self, front_mask, side_mask, front_lm, side_lm,
                     front_scale, side_scale) -> float:
        """
        Measure calf circumference at widest point (typically 1/3 from knee down to ankle)
        """
        if not front_lm or 25 not in front_lm or 27 not in front_lm:
            return 36.0  # Average calf circumference
        
        h_front = front_mask.shape[0]
        h_side = side_mask.shape[0]
        w_side = side_mask.shape[1]
        
        left_knee = front_lm[25]
        left_ankle = front_lm[27]
        
        # Calf measurement at 1/3 distance from knee to ankle (widest part)
        calf_y = int(left_knee[1] * 0.67 + left_ankle[1] * 0.33)
        calf_x = left_knee[0]
        
        # Measure width from front view
        _, _, calf_width_px = self._get_contiguous_width(front_mask, calf_y, calf_x, search_range=8)
        
        if calf_width_px == 0:
            return 36.0
        
        calf_width_cm = calf_width_px * front_scale
        
        # Measure depth from side view
        if side_lm and 25 in side_lm and 27 in side_lm:
            side_knee = side_lm[25]
            side_ankle = side_lm[27]
            side_calf_y = int(side_knee[1] * 0.67 + side_ankle[1] * 0.33)
            side_calf_x = side_knee[0]
        else:
            # Estimate position if side landmarks unavailable
            side_calf_y = int(h_side * (calf_y / h_front))
            side_calf_x = w_side // 2
        
        _, _, calf_depth_px = self._get_contiguous_width(side_mask, side_calf_y, side_calf_x, search_range=8)
        
        if calf_depth_px == 0:
            calf_depth_cm = calf_width_cm * 0.90
        else:
            calf_depth_cm = calf_depth_px * side_scale
        
        # Calculate elliptical circumference
        a = calf_width_cm / 2
        b = calf_depth_cm / 2
        h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0
        calf_circumference = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        
        return calf_circumference
    
    def _calculate_circumferences(self, front_mask, side_mask, back_mask,
                                  front_lm, side_lm, back_lm,
                                  front_scale, side_scale, back_scale) -> BodyMeasurements:
        
        def ellipse_circumference(width_cm: float, depth_cm: float) -> float:
            a, b = width_cm / 2, depth_cm / 2
            if a == 0 or b == 0:
                return 0
            h = ((a - b) ** 2) / ((a + b) ** 2)
            return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        
        h_front = front_mask.shape[0]
        w_front = front_mask.shape[1]
        h_side = side_mask.shape[0]
        w_side = side_mask.shape[1]
        
        left_shoulder = front_lm.get(11) if front_lm else None
        right_shoulder = front_lm.get(12) if front_lm else None
        left_hip = front_lm.get(23) if front_lm else None
        right_hip = front_lm.get(24) if front_lm else None
        left_knee = front_lm.get(25) if front_lm else None
        
        if left_shoulder and right_shoulder:
            torso_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        else:
            torso_center_x = w_front // 2
        
        # NECK
        neck_width_cm, neck_depth_cm, neck = self._measure_neck(
            front_mask, side_mask, front_lm, side_lm, front_scale, side_scale
        )
        
        # SHOULDER WIDTH (bone-to-bone)
        shoulder_width = self._measure_shoulder_width(front_lm, front_scale)
        
        # ABDOMEN
        if left_shoulder and left_hip:
            abdomen_y = int(left_shoulder[1] * 0.35 + left_hip[1] * 0.65)
            x1_f, x2_f, w_f = self._measure_torso_width(front_mask, abdomen_y, torso_center_x)
            x1_b, x2_b, w_b = self._measure_torso_width(back_mask, abdomen_y, torso_center_x)
            abdomen_width_cm = ((w_f + w_b) / 2) * front_scale
            
            _, _, abdomen_depth_px = self._get_contiguous_width(side_mask, int(h_side * 0.42), w_side // 2)
            abdomen_depth_cm = abdomen_depth_px * side_scale
            abdomen = ellipse_circumference(abdomen_width_cm, abdomen_depth_cm)
        else:
            abdomen = 82.0
        
        # HIP
        if left_hip and right_hip:
            hip_y = (left_hip[1] + right_hip[1]) // 2
            x1_f, x2_f, w_f = self._measure_torso_width(front_mask, hip_y, torso_center_x)
            x1_b, x2_b, w_b = self._measure_torso_width(back_mask, hip_y, torso_center_x)
            hip_width_cm = ((w_f + w_b) / 2) * front_scale
            
            _, _, hip_depth_px = self._get_contiguous_width(side_mask, int(h_side * 0.55), w_side // 2)
            hip_depth_cm = hip_depth_px * side_scale
            hip = ellipse_circumference(hip_width_cm, hip_depth_cm)
        else:
            hip = 98.0
        
        # THIGH
        if left_hip and left_knee:
            thigh_y = (left_hip[1] + left_knee[1]) // 2
            thigh_x = left_hip[0]
            _, _, thigh_width_px = self._get_contiguous_width(front_mask, thigh_y, thigh_x)
            thigh_width_cm = thigh_width_px * front_scale
            
            _, _, thigh_depth_px = self._get_contiguous_width(side_mask, int(h_side * 0.63), w_side // 2)
            thigh_depth_cm = thigh_depth_px * side_scale
            thigh = ellipse_circumference(thigh_width_cm, thigh_depth_cm)
        else:
            thigh = 52.0
        
        # KNEE
        if left_knee:
            knee_y = left_knee[1]
            knee_x = left_knee[0]
            _, _, knee_width_px = self._get_contiguous_width(front_mask, knee_y, knee_x)
            knee_width_cm = knee_width_px * front_scale
            
            if side_lm and 25 in side_lm:
                side_knee_y = side_lm[25][1]
                side_knee_x = side_lm[25][0]
            else:
                side_knee_y = int(h_side * (knee_y / h_front))
                side_knee_x = w_side // 2
            
            _, _, knee_depth_px = self._get_contiguous_width(side_mask, side_knee_y, side_knee_x, search_range=10)
            
            if knee_depth_px == 0:
                knee_depth_cm = knee_width_cm * 0.85
            else:
                knee_depth_cm = knee_depth_px * side_scale
            
            knee = ellipse_circumference(knee_width_cm, knee_depth_cm)
        else:
            knee = 36.0
        
        # CALF
        calf = self._measure_calf(front_mask, side_mask, front_lm, side_lm,
                                  front_scale, side_scale)
        
        # ANKLE (constant value - could be improved)
        ankle = 23.0
        
        return BodyMeasurements(
            neck=round(neck, 1),
            shoulder_width=round(shoulder_width, 1),
            abdomen=round(abdomen, 1),
            hip=round(hip, 1),
            thigh=round(thigh, 1),
            knee=round(knee, 1),
            calf=round(calf, 1),
            ankle=round(ankle, 1)
        )
    
    def _visualize(self, front_img, side_img, back_img,
                   front_mask, side_mask, back_mask,
                   front_lm, side_lm, back_lm,
                   measurements):
        
        colors = {
            'neck': (255, 100, 100),
            'shoulder': (100, 150, 255),
            'abdomen': (100, 255, 255),
            'hip': (255, 100, 255),
            'thigh': (255, 180, 100),
            'knee': (180, 100, 255),
            'calf': (255, 200, 150)
        }
        
        def draw_landmarks(img, lm):
            if not lm:
                return
            for idx, (x, y, v) in lm.items():
                if v > 0.5:
                    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        
        def draw_line(img, y, x1, x2, label, color):
            if x1 >= x2:
                return
            cv2.line(img, (x1, y), (x2, y), color, 3)
            cv2.circle(img, (x1, y), 6, color, -1)
            cv2.circle(img, (x2, y), 6, color, -1)
            
            cv2.putText(img, label, (x1, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, label, (x1, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        front_vis = cv2.addWeighted(front_img, 0.7, cv2.cvtColor(front_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        side_vis = cv2.addWeighted(side_img, 0.7, cv2.cvtColor(side_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        draw_landmarks(front_vis, front_lm)
        draw_landmarks(side_vis, side_lm)
        
        if front_lm:
            nose = front_lm.get(0)
            left_shoulder = front_lm.get(11)
            right_shoulder = front_lm.get(12)
            left_hip = front_lm.get(23)
            left_knee = front_lm.get(25)
            left_ankle = front_lm.get(27)
            
            torso_center_x = (left_shoulder[0] + right_shoulder[0]) // 2 if left_shoulder and right_shoulder else front_mask.shape[1] // 2
            
            # Shoulder width (bone-to-bone)
            if left_shoulder and right_shoulder:
                cv2.line(front_vis, (left_shoulder[0], left_shoulder[1]), 
                        (right_shoulder[0], right_shoulder[1]), colors['shoulder'], 3)
                cv2.circle(front_vis, (left_shoulder[0], left_shoulder[1]), 8, colors['shoulder'], -1)
                cv2.circle(front_vis, (right_shoulder[0], right_shoulder[1]), 8, colors['shoulder'], -1)
                
                mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
                mid_y = (left_shoulder[1] + right_shoulder[1]) // 2
                label = f"Shoulder {measurements.shoulder_width}cm"
                cv2.putText(front_vis, label, (mid_x - 80, mid_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(front_vis, label, (mid_x - 80, mid_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['shoulder'], 2)
            
            # Neck
            if nose and left_shoulder and right_shoulder:
                shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
                shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) // 2
                neck_x = (nose[0] + shoulder_mid_x) // 2
                neck_y = (nose[1] + shoulder_mid_y) // 2
                
                x1, x2, _ = self._get_contiguous_width(front_mask, neck_y, neck_x, search_range=5)
                draw_line(front_vis, neck_y, x1, x2, f"Neck {measurements.neck}cm", colors['neck'])
            
            # Abdomen
            if left_shoulder and left_hip:
                abdomen_y = int(left_shoulder[1] * 0.35 + left_hip[1] * 0.65)
                x1, x2, _ = self._measure_torso_width(front_mask, abdomen_y, torso_center_x)
                draw_line(front_vis, abdomen_y, x1, x2, f"Abdomen {measurements.abdomen}cm", colors['abdomen'])
            
            # Hip
            if left_hip:
                hip_y = left_hip[1]
                x1, x2, _ = self._measure_torso_width(front_mask, hip_y, torso_center_x)
                draw_line(front_vis, hip_y, x1, x2, f"Hip {measurements.hip}cm", colors['hip'])
            
            # Thigh
            if left_hip and left_knee:
                thigh_y = (left_hip[1] + left_knee[1]) // 2
                thigh_x = left_hip[0]
                x1, x2, _ = self._get_contiguous_width(front_mask, thigh_y, thigh_x)
                draw_line(front_vis, thigh_y, x1, x2, f"Thigh {measurements.thigh}cm", colors['thigh'])
            
            # Knee
            if left_knee:
                knee_y = left_knee[1]
                knee_x = left_knee[0]
                x1, x2, _ = self._get_contiguous_width(front_mask, knee_y, knee_x)
                draw_line(front_vis, knee_y, x1, x2, f"Knee {measurements.knee}cm", colors['knee'])
            
            # Calf
            if left_knee and left_ankle:
                calf_y = int(left_knee[1] * 0.67 + left_ankle[1] * 0.33)
                calf_x = left_knee[0]
                x1, x2, _ = self._get_contiguous_width(front_mask, calf_y, calf_x, search_range=8)
                draw_line(front_vis, calf_y, x1, x2, f"Calf {measurements.calf}cm", colors['calf'])
        
        for img, title in [(front_vis, "FRONT"), (side_vis, "SIDE")]:
            cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imwrite("front_measurements.jpg", front_vis)
        cv2.imwrite("side_measurements.jpg", side_vis)
        
        print("\n✓ Visualizations saved")


if __name__ == "__main__":
    estimator = BodyCircumferenceEstimator(use_yolo=False)
    
    measurements = estimator.estimate(
        front_img_path="front_view.png",
        side_img_path="side_view.png",
        back_img_path="back_view.png",
        height_cm=178.0
    )
    
    print("\n" + "="*50)
    print("Body Measurements (cm):")
    print("="*50)
    print(f"Neck:           {measurements.neck}")
    print(f"Shoulder Width: {measurements.shoulder_width}")
    print(f"Abdomen:        {measurements.abdomen}")
    print(f"Hip:            {measurements.hip}")
    print(f"Thigh:          {measurements.thigh}")
    print(f"Knee:           {measurements.knee}")
    print(f"Calf:           {measurements.calf}")
    print(f"Ankle:          {measurements.ankle}")
    print("="*50)