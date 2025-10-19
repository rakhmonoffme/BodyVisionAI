'use client';

import { create } from 'zustand';
import { Photos, UserInfo, ProcessingState, AnalysisResult, PhotoType } from '@/types';
import { createAnalysisSession, simulateAnalysis } from '@/lib/api';

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
    const { userInfo } = get();

    set((state) => ({
      processing: {
        ...state.processing,
        status: 'uploading',
        progress: 0,
        currentStep: 'Creating session...',
      },
    }));

    try {
      const sessionId = await createAnalysisSession(userInfo);

      set((state) => ({
        processing: {
          ...state.processing,
          sessionId,
          status: 'processing',
          currentStep: 'Starting analysis...',
        },
      }));

      await simulateAnalysis(sessionId);

      set((state) => ({
        processing: {
          ...state.processing,
          status: 'completed',
          progress: 100,
        },
      }));
    } catch (error) {
      console.error('Analysis failed:', error);
      set((state) => ({
        processing: {
          ...state.processing,
          status: 'error',
          currentStep: 'Analysis failed. Please try again.',
        },
      }));
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
