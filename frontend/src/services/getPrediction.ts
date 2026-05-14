import axios from 'axios';
import { PredictionData } from '../store/useStore';

export const getPrediction = async (file: File, modelType: string): Promise<PredictionData> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_type', modelType);

  try {
    const response = await axios.post<PredictionData>('http://localhost:8000/api/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching prediction:', error);
    throw error;
  }
};
