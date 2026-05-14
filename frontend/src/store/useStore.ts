import { create } from 'zustand';

interface AngleInfo {
  angle: number;
  idxs: [number, number];
}

export interface PredictionData {
  detections: number[][]; // [x1, y1, x2, y2][]
  landmarks: number[][];  // [x, y][] (4 per vertebra)
  cobb_angles_list: number[];
  angles_with_pos: {
    pt: AngleInfo;
    mt: AngleInfo;
    lt: AngleInfo;
  };
  curve_type: string;
  midpoint_lines: number[][][]; // [[x1,y1], [x2,y2]][]
}

interface AppState {
  imageFile: File | null;
  imageUrl: string | null;
  selectedModel: string;
  prediction: PredictionData | null;
  isLoading: boolean;
  
  // Visualization toggles
  displayDetections: boolean;
  showDetectionLabels: boolean;
  detectionScale: number;
  
  displayKeypoints: boolean;
  keypointMode: 'dots' | 'horizontal' | 'connected' | 'quadrilateral';
  keypointColor: string;
  keypointRadius: number;
  
  displayCobbAngle: boolean;

  // Actions
  setImage: (file: File | null, url: string | null) => void;
  setSelectedModel: (model: string) => void;
  setPrediction: (data: PredictionData | null) => void;
  setIsLoading: (loading: boolean) => void;
  
  setDisplayDetections: (val: boolean) => void;
  setShowDetectionLabels: (val: boolean) => void;
  setDetectionScale: (val: number) => void;
  setDisplayKeypoints: (val: boolean) => void;
  setKeypointMode: (mode: 'dots' | 'horizontal' | 'connected' | 'quadrilateral') => void;
  setKeypointColor: (color: string) => void;
  setKeypointRadius: (val: number) => void;
  setDisplayCobbAngle: (val: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  imageFile: null,
  imageUrl: null,
  selectedModel: 'keypoint_rcnn',
  prediction: null,
  isLoading: false,
  
  displayDetections: true,
  showDetectionLabels: false,
  detectionScale: 2,
  
  displayKeypoints: true,
  keypointMode: 'quadrilateral',
  keypointColor: '#00ffff',
  keypointRadius: 3,
  
  displayCobbAngle: true,

  setImage: (file, url) => set({ imageFile: file, imageUrl: url, prediction: null }),
  setSelectedModel: (model) => set({ selectedModel: model, prediction: null }),
  setPrediction: (data) => set({ prediction: data }),
  setIsLoading: (loading) => set({ isLoading: loading }),
  
  setDisplayDetections: (val) => set({ displayDetections: val }),
  setShowDetectionLabels: (val) => set({ showDetectionLabels: val }),
  setDetectionScale: (val) => set({ detectionScale: val }),
  setDisplayKeypoints: (val) => set({ displayKeypoints: val }),
  setKeypointMode: (mode) => set({ keypointMode: mode }),
  setKeypointColor: (color) => set({ keypointColor: color }),
  setKeypointRadius: (val) => set({ keypointRadius: val }),
  setDisplayCobbAngle: (val) => set({ displayCobbAngle: val }),
}));
