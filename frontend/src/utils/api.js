// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/predict`,
  HEALTH: `${API_BASE_URL}/health`,
  MODEL_INFO: `${API_BASE_URL}/model-info`,
  HISTORICAL_DATA: `${API_BASE_URL}/historical-data`,
};

export default API_BASE_URL;
