export type Gender = 'male' | 'female' | 'neutral';
export type PhotoType = 'front' | 'side' | 'back';
export type AnalysisStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';

export interface UserInfo {
  height: number;
  weight: number;
  age: number;
  gender: Gender;
}

export interface Photos {
  front: File | null;
  side: File | null;
  back: File | null;
}

export interface ProcessingState {
  sessionId: string | null;
  backendSessionId: string | null;
  status: AnalysisStatus;
  progress: number;
  currentStep: string;
}

// Backend response types
export interface BodyComposition {
  body_fat_percentage: number;
  fat_mass_kg: number;
  lean_mass_kg: number;
  muscle_mass_kg: number;
  bmi: number;
  volume_liters: number;
  body_density: number;
}

export interface HealthMetrics {
  waist_to_hip_ratio: number;
  health_status: {
    category: 'Athletic' | 'Fit' | 'Acceptable' | 'Obese';
    risk_level: 'Low' | 'Moderate' | 'High';
    recommendation: string;
  };
}

export interface AnalysisResult {
  sessionId: string;
  status: string;
  timestamp: string;
  measurements: {
    neck: number;
    chest: number;
    abdomen: number;
    waist: number;
    hip: number;
    thigh: number;
    knee: number;
    ankle: number;
    shoulder_width: number;
    calf: number;
  };
  bodyComposition: BodyComposition;
  healthMetrics: HealthMetrics;
  images: {
    front: string;
    side: string;
    back: string;
    mesh: string;
  };
}