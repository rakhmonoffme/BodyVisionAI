'use client';

import { create } from 'zustand';
import { Photos, UserInfo, ProcessingState, AnalysisResult, PhotoType } from '@/types';
import { submitAnalysis } from '@/lib/api';

interface AppStore {
  photos: Photos;
  userInfo: UserInfo;
  processing: ProcessingState;
  results: AnalysisResult | null;
  setPhoto: (type: PhotoType, file: File) => void;
  setUserInfo: (info: Partial<UserInfo>) => void;
  submitAnalysis: () => Promise<void>;
  resetAnalysis: () => void;
}

const initialUserInfo: UserInfo = {
  height: 170,
  weight: 70,
  age: 30,
  gender: 'neutral',
};

const initialProcessing: ProcessingState = {
  sessionId: null,
  backendSessionId: null,
  status: 'idle',
  progress: 0,
  currentStep: '',
};

export const useAppStore = create<AppStore>((set, get) => ({
  photos: {
    front: null,
    side: null,
    back: null,
  },
  userInfo: initialUserInfo,
  processing: initialProcessing,
  results: null,

  setPhoto: (type: PhotoType, file: File) => {
    set((state) => ({
      photos: {
        ...state.photos,
        [type]: file,
      },
    }));
  },

  setUserInfo: (info: Partial<UserInfo>) => {
    set((state) => ({
      userInfo: {
        ...state.userInfo,
        ...info,
      },
    }));
  },

  submitAnalysis: async () => {
    const { photos, userInfo } = get();
    
    if (!photos.front || !photos.side || !photos.back) {
      throw new Error('All photos required');
    }
    
    set({ 
      processing: { 
        sessionId: null,
        backendSessionId: null,
        status: 'processing', 
        progress: 0, 
        currentStep: 'Uploading photos...' 
      } 
    });
    
    // Simulate progress stages
    const progressStages = [
      { delay: 3000, progress: 15, step: 'Processing front view...' },
      { delay: 6000, progress: 30, step: 'Processing side view...' },
      { delay: 9000, progress: 45, step: 'Processing back view...' },
      { delay: 15000, progress: 60, step: 'Extracting body measurements...' },
      { delay: 25000, progress: 75, step: 'Generating 3D body model...' },
      { delay: 35000, progress: 90, step: 'Calculating body composition...' },
    ];
    
    // Set up progress simulation
    const timers: NodeJS.Timeout[] = [];
    progressStages.forEach(({ delay, progress, step }) => {
      const timer = setTimeout(() => {
        const currentState = get().processing;
        if (currentState.status === 'processing') {
          set({
            processing: {
              ...currentState,
              progress,
              currentStep: step,
            },
          });
        }
      }, delay);
      timers.push(timer);
    });
    
    try {
      // Direct call to Flask backend (30-60 seconds)
      const result = await submitAnalysis(
        photos as { front: File; side: File; back: File },
        userInfo
      );
      
      // Clear all timers
      timers.forEach(timer => clearTimeout(timer));
      
      // Store results immediately
      set({
        processing: {
          sessionId: result.sessionId,
          backendSessionId: result.sessionId,
          status: 'completed',
          progress: 100,
          currentStep: 'Analysis complete!',
        },
        results: result,
      });
    } catch (error) {
      // Clear all timers
      timers.forEach(timer => clearTimeout(timer));
      
      set({
        processing: {
          sessionId: null,
          backendSessionId: null,
          status: 'error',
          progress: 0,
          currentStep: error instanceof Error ? error.message : 'Analysis failed',
        },
      });
      throw error;
    }
  },

  resetAnalysis: () => {
    set({
      photos: {
        front: null,
        side: null,
        back: null,
      },
      userInfo: initialUserInfo,
      processing: initialProcessing,
      results: null,
    });
  },
}));