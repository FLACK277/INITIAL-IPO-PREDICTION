"""
FastAPI Backend for IPO Prediction System
Replicates exact feature engineering from ML model working.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import joblib
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IPO Predict AI API",
    description="Backend API for IPO prediction using ML ensemble models",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}
scaler = None
selected_features = None
models_loaded = False

# Model file paths
MODEL_DIR = Path(__file__).parent.parent / "model"
SCALER_PATH = MODEL_DIR / "ipo_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "ipo_selected_features.pkl"
ENSEMBLE_MODEL_PATH = MODEL_DIR / "ipo_model_ensemble.pkl"
NEURAL_NETWORK_PATH = MODEL_DIR / "ipo_model_neural_network.h5"
CSV_PATH = Path(__file__).parent.parent / "Indian_IPO_Market_Data.csv"


# Pydantic models for request/response
class IPOInput(BaseModel):
    """Input model for IPO prediction"""
    ipo_name: str = Field(..., description="Name of the IPO")
    issue_size: float = Field(..., description="Issue size in crores", gt=0)
    issue_price: float = Field(..., description="Issue price in INR", gt=0)
    subscription_qib: float = Field(..., description="QIB subscription in times", ge=0)
    subscription_hni: float = Field(..., description="HNI subscription in times", ge=0)
    subscription_rii: float = Field(..., description="RII/Retail subscription in times", ge=0)
    subscription_total: float = Field(..., description="Total subscription in times", ge=0)


class ModelResult(BaseModel):
    """Result from a single model"""
    model: str
    probability: float
    prediction: str


class PredictionResponse(BaseModel):
    """Response model for IPO prediction"""
    ipo_name: str
    prediction: str
    probability: float
    confidence: float
    risk_level: str
    predicted_listing_gain_percent: float
    predicted_opening_price: float
    recommendation: str
    model_results: List[ModelResult]
    warning: Optional[str] = None


def create_features(data: Dict[str, float]) -> Dict[str, float]:
    """
    Create engineered features - EXACT replication from ML model working.py
    This is critical for model compatibility
    """
    engineered_data = data.copy()
    
    # Interaction features
    engineered_data['QIB_RII_Interaction'] = data['Issue_Size'] * data['Subscription_RII']
    engineered_data['HNI_RII_Interaction'] = data['Subscription_HNI'] * data['Subscription_RII']
    engineered_data['QIB_HNI_Interaction'] = data['Subscription_QIB'] * data['Subscription_HNI']
    engineered_data['Size_Price_Ratio'] = data['Issue_Size'] / (data['Issue_Price'] + 1e-8)
    engineered_data['Subscription_Imbalance'] = abs(data['Subscription_QIB'] - data['Subscription_RII'])
    engineered_data['Price_QIB_Interaction'] = data['Issue_Price'] * data['Subscription_QIB']
    engineered_data['Size_QIB_Interaction'] = data['Issue_Size'] * data['Subscription_QIB']
    
    # Logarithmic transformations for highly skewed features
    for col in ['Issue_Size', 'Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Subscription_Total']:
        engineered_data[f'Log_{col}'] = np.log1p(data[col])
    
    # Polynomial features
    engineered_data['Issue_Size_Squared'] = data['Issue_Size'] ** 2
    engineered_data['Issue_Price_Squared'] = data['Issue_Price'] ** 2
    
    # Statistical features
    sub_values = [data['Subscription_QIB'], data['Subscription_HNI'], data['Subscription_RII']]
    engineered_data['Subscription_Mean'] = np.mean(sub_values)
    engineered_data['Subscription_Std'] = np.std(sub_values)
    engineered_data['Subscription_Range'] = max(sub_values) - min(sub_values)
    engineered_data['QIB_to_RII_Ratio'] = data['Subscription_QIB'] / (data['Subscription_RII'] + 1e-8)
    engineered_data['HNI_to_RII_Ratio'] = data['Subscription_HNI'] / (data['Subscription_RII'] + 1e-8)
    
    # Composite features
    engineered_data['Weighted_Subscription'] = (
        data['Subscription_QIB'] * 0.5 + 
        data['Subscription_HNI'] * 0.3 + 
        data['Subscription_RII'] * 0.2
    )
    
    return engineered_data


def predict_opening_price(
    issue_price: float,
    subscription_metrics: Dict[str, float],
    avg_probability: float,
    consensus: str
) -> tuple:
    """
    Predict opening price - EXACT replication from ML model working.py
    """
    # Base parameters for gain calculation
    base_gain = 10  # Minimum expected gain for profitable IPOs
    max_gain = 50   # Maximum expected gain for highly subscribed IPOs
    base_loss = -5  # Minimum expected loss for unprofitable IPOs
    max_loss = -20  # Maximum expected loss for poorly subscribed IPOs
    
    # Calculate subscription strength score (0-1)
    qib_weight, hni_weight, rii_weight = 0.5, 0.3, 0.2
    
    # Normalize subscription values using log scale to handle extreme values
    qib_norm = min(1.0, np.log1p(subscription_metrics['Subscription_QIB']) / np.log1p(100))
    hni_norm = min(1.0, np.log1p(subscription_metrics['Subscription_HNI']) / np.log1p(100))
    rii_norm = min(1.0, np.log1p(subscription_metrics['Subscription_RII']) / np.log1p(50))
    total_norm = min(1.0, np.log1p(subscription_metrics['Subscription_Total']) / np.log1p(100))
    
    # Weighted subscription score
    subscription_score = (
        qib_norm * qib_weight + 
        hni_norm * hni_weight + 
        rii_norm * rii_weight
    )
    
    # Boost subscription score based on total subscription
    subscription_score = 0.7 * subscription_score + 0.3 * total_norm
    
    # Calculate expected gain/loss percentage
    if consensus == "PROFITABLE":
        # For profitable predictions, scale between base_gain and max_gain
        confidence_factor = (avg_probability - 0.5) * 2  # Map from [0.5, 1.0] to [0, 1.0]
        
        # Combine confidence and subscription score with non-linear scaling
        combined_score = 0.6 * confidence_factor + 0.4 * subscription_score
        combined_score = combined_score ** 0.8  # Non-linear scaling
        
        listing_gain_percent = base_gain + (max_gain - base_gain) * combined_score
        
        # Apply price dampening for higher priced IPOs
        price_dampening = max(0.8, 1 - (issue_price / 1000) * 0.1)
        listing_gain_percent *= price_dampening
    else:
        # For unprofitable predictions
        confidence_factor = (0.5 - avg_probability) * 2
        combined_score = 0.7 * confidence_factor + 0.3 * (1 - subscription_score)
        listing_gain_percent = base_loss - (max_loss - base_loss) * combined_score
    
    # Calculate predicted opening price
    predicted_opening_price = issue_price * (1 + listing_gain_percent / 100)
    
    return round(predicted_opening_price, 2), round(listing_gain_percent, 2)


def get_risk_level(probability: float, prediction: str) -> str:
    """Determine risk level based on probability and prediction"""
    if prediction == "PROFITABLE":
        if probability > 0.7:
            return "Low"
        elif probability > 0.6:
            return "Medium"
        else:
            return "Medium-High"
    else:
        if probability < 0.3:
            return "High"
        elif probability < 0.4:
            return "Medium-High"
        else:
            return "Medium"


def get_recommendation(consensus: str, confidence: float, listing_gain: float) -> str:
    """Generate investment recommendation"""
    if consensus == "PROFITABLE":
        if confidence > 70 and listing_gain > 20:
            return "Strong BUY"
        elif confidence > 60 and listing_gain > 10:
            return "Moderate BUY"
        elif listing_gain > 5:
            return "Cautious BUY"
        else:
            return "HOLD"
    else:
        if confidence > 70:
            return "Strong AVOID"
        else:
            return "AVOID"


def load_models_from_disk():
    """Load trained models from disk"""
    global models, scaler, selected_features, models_loaded
    
    try:
        logger.info("Loading models from disk...")
        
        # Load scaler
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            logger.info("✓ Scaler loaded")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH}")
            return False
        
        # Load selected features
        if FEATURES_PATH.exists():
            selected_features = joblib.load(FEATURES_PATH)
            logger.info(f"✓ Selected features loaded ({len(selected_features)} features)")
        else:
            logger.warning(f"Selected features not found at {FEATURES_PATH}")
            return False
        
        # Load ensemble model
        if ENSEMBLE_MODEL_PATH.exists():
            models['Ensemble'] = joblib.load(ENSEMBLE_MODEL_PATH)
            logger.info("✓ Ensemble model loaded")
        
        # Load neural network model
        if NEURAL_NETWORK_PATH.exists():
            try:
                import tensorflow as tf
                from tensorflow import keras
                models['Neural Network'] = keras.models.load_model(NEURAL_NETWORK_PATH)
                # Warm up the model
                dummy_input = np.zeros((1, len(selected_features)))
                _ = models['Neural Network'].predict(dummy_input, verbose=0)
                logger.info("✓ Neural Network model loaded")
            except Exception as e:
                logger.error(f"Error loading neural network: {e}")
        
        if models:
            models_loaded = True
            logger.info(f"✓ Successfully loaded {len(models)} model(s)")
            return True
        else:
            logger.warning("No models loaded")
            return False
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


def create_mock_prediction(ipo_input: IPOInput) -> PredictionResponse:
    """Create mock prediction when models are not available"""
    # Simple heuristic for mock prediction
    avg_subscription = (
        ipo_input.subscription_qib * 0.5 + 
        ipo_input.subscription_hni * 0.3 + 
        ipo_input.subscription_rii * 0.2
    )
    
    if avg_subscription > 5:
        probability = 0.75
        prediction = "PROFITABLE"
        listing_gain = 15.0
    elif avg_subscription > 2:
        probability = 0.65
        prediction = "PROFITABLE"
        listing_gain = 8.0
    else:
        probability = 0.35
        prediction = "NOT PROFITABLE"
        listing_gain = -5.0
    
    confidence = max(probability, 1 - probability) * 100
    opening_price = ipo_input.issue_price * (1 + listing_gain / 100)
    
    return PredictionResponse(
        ipo_name=ipo_input.ipo_name,
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        risk_level=get_risk_level(probability, prediction),
        predicted_listing_gain_percent=listing_gain,
        predicted_opening_price=round(opening_price, 2),
        recommendation=get_recommendation(prediction, confidence, listing_gain),
        model_results=[
            ModelResult(model="Mock Model", probability=probability, prediction=prediction)
        ],
        warning="⚠️ Using mock predictions. Model files not found. Please place trained models in /model directory."
    )


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting IPO Predict AI API...")
    load_models_from_disk()
    if not models_loaded:
        logger.warning("⚠️ Models not loaded. API will use mock predictions.")
    else:
        logger.info("✓ API ready with trained models")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "IPO Predict AI API",
        "version": "1.0.0",
        "status": "online",
        "models_loaded": models_loaded
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "available_models": list(models.keys()) if models_loaded else []
    }


@app.get("/model-info")
async def model_info():
    """Get information about loaded models"""
    if not models_loaded:
        return {
            "models_loaded": False,
            "message": "Models not loaded. Using mock predictions.",
            "expected_files": [
                str(SCALER_PATH),
                str(FEATURES_PATH),
                str(ENSEMBLE_MODEL_PATH),
                str(NEURAL_NETWORK_PATH)
            ]
        }
    
    return {
        "models_loaded": True,
        "available_models": list(models.keys()),
        "num_features": len(selected_features) if selected_features else 0,
        "feature_names": selected_features.tolist() if selected_features is not None and hasattr(selected_features, 'tolist') else selected_features,
        "model_performance": {
            "Logistic Regression": {"auc": 0.7025, "accuracy": 0.656},
            "Random Forest": {"auc": 0.6837, "accuracy": 0.688},
            "SVM": {"auc": 0.6562, "accuracy": 0.641},
            "Voting Ensemble": {"auc": 0.6611, "accuracy": 0.672},
            "Neural Network": {"auc": 0.90, "accuracy": 0.740}
        }
    }


@app.get("/historical-data")
async def get_historical_data():
    """Get historical IPO data from CSV"""
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="Historical data CSV not found")
        
        df = pd.read_csv(CSV_PATH)
        
        # Convert to JSON-friendly format
        data = df.to_dict(orient='records')
        
        return {
            "count": len(data),
            "data": data
        }
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_ipo(ipo_input: IPOInput):
    """
    Predict IPO listing performance
    
    Returns prediction, probability, confidence, risk level, and investment recommendation
    """
    try:
        # If models not loaded, return mock prediction
        if not models_loaded:
            logger.warning("Models not loaded, returning mock prediction")
            return create_mock_prediction(ipo_input)
        
        # Prepare input data
        input_data = {
            'Issue_Size': ipo_input.issue_size,
            'Issue_Price': ipo_input.issue_price,
            'Subscription_QIB': ipo_input.subscription_qib,
            'Subscription_HNI': ipo_input.subscription_hni,
            'Subscription_RII': ipo_input.subscription_rii,
            'Subscription_Total': ipo_input.subscription_total
        }
        
        # Create engineered features
        engineered_data = create_features(input_data)
        
        # Filter to only include selected features
        filtered_data = {
            feature: engineered_data.get(feature, 0) 
            for feature in selected_features
        }
        
        # Create DataFrame and scale
        input_df = pd.DataFrame([filtered_data], columns=selected_features)
        input_scaled = scaler.transform(input_df)
        
        # Make predictions with each model
        results = []
        probabilities = []
        
        for model_name, model in models.items():
            try:
                # Neural network models
                if hasattr(model, 'predict') and 'Neural' in model_name:
                    prediction_prob = float(model.predict(input_scaled, verbose=0)[0][0])
                # Traditional ML models
                else:
                    prediction_prob = float(model.predict_proba(input_scaled)[0][1])
                
                prediction = "PROFITABLE" if prediction_prob > 0.5 else "NOT PROFITABLE"
                probabilities.append(prediction_prob)
                
                results.append(ModelResult(
                    model=model_name,
                    probability=round(prediction_prob, 4),
                    prediction=prediction
                ))
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                continue
        
        if not probabilities:
            raise Exception("No models could make predictions")
        
        # Calculate consensus
        avg_probability = sum(probabilities) / len(probabilities)
        consensus = "PROFITABLE" if avg_probability > 0.5 else "NOT PROFITABLE"
        confidence = max(avg_probability, 1 - avg_probability) * 100
        
        # Predict opening price and listing gain
        subscription_metrics = {
            'Subscription_QIB': ipo_input.subscription_qib,
            'Subscription_HNI': ipo_input.subscription_hni,
            'Subscription_RII': ipo_input.subscription_rii,
            'Subscription_Total': ipo_input.subscription_total
        }
        
        predicted_opening_price, listing_gain_percent = predict_opening_price(
            ipo_input.issue_price,
            subscription_metrics,
            avg_probability,
            consensus
        )
        
        # Determine risk level and recommendation
        risk_level = get_risk_level(avg_probability, consensus)
        recommendation = get_recommendation(consensus, confidence, listing_gain_percent)
        
        return PredictionResponse(
            ipo_name=ipo_input.ipo_name,
            prediction=consensus,
            probability=round(avg_probability, 4),
            confidence=round(confidence, 2),
            risk_level=risk_level,
            predicted_listing_gain_percent=listing_gain_percent,
            predicted_opening_price=predicted_opening_price,
            recommendation=recommendation,
            model_results=results
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
