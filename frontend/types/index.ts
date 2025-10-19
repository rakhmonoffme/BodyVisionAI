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
  status: AnalysisStatus;
  progress: number;
  currentStep: string;
}

export interface Measurement {
  id: string;
  session_id: string;
  measurement_type: string;
  value: number;
  reference_min: number | null;
  reference_max: number | null;
  created_at: string;
}

export interface BodyComposition {
  id: string;
  session_id: string;
  body_fat_percentage: number | null;
  lean_mass: number | null;
  fat_mass: number | null;
  bmi: number | null;
  body_type: string | null;
  created_at: string;
}

export interface SessionImage {
  id: string;
  session_id: string;
  image_type: string;
  image_url: string;
  created_at: string;
}

export interface AnalysisResult {
  session: {
    id: string;
    user_height: number;
    user_weight: number;
    user_age: number;
    user_gender: string;
    status: string;
    progress: number;
    current_step: string | null;
    created_at: string;
    updated_at: string;
    completed_at: string | null;
  };
  measurements: Measurement[];
  bodyComposition: BodyComposition | null;
  images: SessionImage[];
}
