# 🚀 IPO Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Scikit--learn-orange)
![React](https://img.shields.io/badge/React-18+-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)

*Predicting IPO profitability using advanced machine learning techniques with a premium 3D web interface*

[Features](#-features) • [Web App](#-web-application) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-machine-learning-models) • [API](#-api-documentation)

</div>

---

## 🌐 Web Application

This project now includes a **premium 3D web application** that provides an intuitive interface for IPO prediction with stunning visualizations!

### Features
- 🎨 **3D Interactive UI** - React Three Fiber powered 3D background with floating geometric shapes
- 💎 **Glassmorphism Design** - Modern frosted glass UI cards with premium fintech aesthetics
- 📊 **Real-time Predictions** - Instant IPO performance analysis with confidence scores
- 📈 **Data Visualizations** - Interactive charts showing model performance and historical data
- 🎯 **Investment Recommendations** - Clear BUY/HOLD/AVOID guidance based on AI analysis
- ⚡ **Fast & Responsive** - Built with Vite for lightning-fast development and production builds

### Tech Stack
- **Frontend**: React 18 + Vite, React Three Fiber, Tailwind CSS, Framer Motion, Recharts
- **Backend**: FastAPI + Uvicorn
- **ML Models**: TensorFlow, Scikit-learn, Ensemble Methods

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Overview

This project addresses the challenging task of predicting Initial Public Offering (IPO) profitability using machine learning techniques. Unlike established stocks, IPOs lack historical trading data, making them particularly difficult to predict. Our solution combines financial metrics, historical IPO data, and advanced machine learning algorithms to provide investors with data-driven insights.

### Key Highlights
- 🎯 **85% Training Accuracy** with Neural Networks
- 📊 **0.7025 AUC Score** achieved by Logistic Regression
- 🔮 **Real-time Predictions** with confidence scores
- 📈 **Investment Recommendations** (BUY/HOLD/SELL)
- 🤖 **Ensemble Methods** for robust predictions

## 🎪 Problem Statement

> *Can machine learning models accurately predict IPO opening prices and listing gains?*

Traditional financial models are inadequate for IPO scenarios as they:
- Rely solely on static company fundamentals
- Don't account for dynamic market sentiment
- Lack historical trading data
- Ignore subscription patterns and investor behavior

Our solution integrates diverse data sources and leverages advanced ML techniques to forecast IPO outcomes.

## ✨ Features

### 🔍 **Advanced Analytics**
- Comprehensive feature engineering with interaction terms
- Multiple outlier detection and handling strategies
- Statistical feature selection using ANOVA and mutual information

### 🤖 **Machine Learning Pipeline**
- Multiple model ensemble (Random Forest, SVM, Neural Networks)
- Bayesian optimization for hyperparameter tuning
- Class imbalance handling with SMOTETomek
- Cross-validation and robust evaluation metrics

### 📊 **Prediction Capabilities**
- Binary classification (Profitable vs Non-profitable)
- Listing gain percentage prediction
- Opening price estimation
- Confidence-based investment recommendations

### 🎨 **Visualization & Reporting**
- ROC curves and confusion matrices
- Feature importance analysis
- Performance metrics visualization
- Comprehensive classification reports

## 📊 Dataset

### Indian IPO Market Data
- **Format**: CSV (Comma-Separated Values)
- **Records**: 326 unique IPOs
- **Features**: 13 attributes per IPO
- **Time Period**: Multiple years of Indian stock market data

### Key Features
| Feature | Description |
|---------|-------------|
| `Company Name` | Name of the company launching IPO |
| `IPO Date` | Date when IPO opened for subscription |
| `Issue Price (Rs)` | Price at which shares were offered |
| `Listing Price (Rs)` | Price at stock exchange listing |
| `Subscription (x)` | Overall subscription multiplier |
| `QIB/NII/Retail` | Subscription by investor categories |
| `Listing Gain (%)` | Target variable for prediction |

## 🤖 Machine Learning Models

### 1. 📈 **Logistic Regression**
- **Best AUC**: 0.7025
- Excellent baseline performance
- Strong feature interpretability

### 2. 🌳 **Random Forest**
- **AUC**: 0.6837
- Handles non-linear relationships
- Robust against overfitting
- Feature importance insights

### 3. ⚡ **Gradient Boosting**
- Sequential learning approach
- Corrects previous model errors
- Captures incremental patterns

### 4. 🎯 **Support Vector Machine**
- Complex boundary modeling
- Kernel-based transformations
- Non-linear relationship capture

### 5. 🗳️ **Voting Ensemble**
- Combines multiple model strengths
- Improved generalization
- Reduced prediction variance

### 6. 🧠 **Neural Network**
- **Architecture**: 2 hidden layers (384, 128 units)
- **Training Accuracy**: 85%
- **Validation Accuracy**: 74%
- Advanced regularization (Dropout, L2)
- Class weight balancing

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
Node.js 18+
npm or yarn package manager
```

### Clone Repository
```bash
git clone https://github.com/FLACK277/INITIAL-IPO-PREDICTION.git
cd INITIAL-IPO-PREDICTION
```

## 🌐 Web Application Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Place trained model files in the `/model` directory:
   - `ipo_scaler.pkl`
   - `ipo_selected_features.pkl`
   - `ipo_model_ensemble.pkl`
   - `ipo_model_neural_network.h5`

   To generate these files, run the training script from the root directory:
   ```bash
   cd ..
   python "ML model working.py"
   # Then move the generated files to the model directory
   mv ipo_*.pkl ipo_*.h5 model/
   ```

4. Start the backend server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The web application will open at `http://localhost:5173`

### Production Build

To build the frontend for production:
```bash
cd frontend
npm run build
```

The optimized build will be in the `dist` directory.

## 💻 Usage

### Web Application

1. **Start the Backend**: Open a terminal and run:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start the Frontend**: Open another terminal and run:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the Application**: Open your browser and navigate to `http://localhost:5173`

4. **Make Predictions**:
   - Navigate to the "Predict" section
   - Fill in the IPO details (issue size, price, subscription data)
   - Click "Predict IPO Performance"
   - View the detailed results including:
     - Predicted profitability (PROFITABLE/NOT PROFITABLE)
     - Confidence score
     - Risk level assessment
     - Expected listing gains
     - Predicted opening price
     - Investment recommendation

### Command Line (Original)
```bash
pip install -r requirements.txt
```

### Required Libraries
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
scipy>=1.7.0
```

## 💻 Usage

### Quick Start
```python
from ipo_predictor import IPOPredictor

# Initialize the predictor
predictor = IPOPredictor()

# Load and preprocess data
predictor.load_data('data/ipo_data.csv')

# Train models
predictor.train_models()

# Make predictions
prediction = predictor.predict({
    'Issue_Price': 45.0,
    'Issue_Size': 1000.0,
    'QIB_Subscription': 2.5,
    'Retail_Subscription': 1.8,
    # ... other features
})

print(f"Prediction: {prediction['recommendation']}")
print(f"Confidence: {prediction['confidence']:.2%}")
print(f"Expected Gain: {prediction['expected_gain']:.2%}")
```

### Training Pipeline
```python
# Complete training pipeline
python src/train_model.py --data data/ipo_data.csv --output models/

# Hyperparameter tuning
python src/optimize_hyperparameters.py --trials 200

# Evaluate models
python src/evaluate_models.py --model_path models/best_model.pkl
```

### Making Predictions
```python
# Single prediction
python src/predict.py --input single_ipo.json

# Batch predictions
python src/predict.py --input batch_ipos.csv --output predictions.csv
```

## 📡 API Documentation

The backend provides a RESTful API for IPO prediction. The API is built with FastAPI and includes automatic interactive documentation.

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["Ensemble", "Neural Network"]
}
```

#### 2. Model Information
```http
GET /model-info
```

**Response:**
```json
{
  "models_loaded": true,
  "available_models": ["Ensemble", "Neural Network"],
  "num_features": 28,
  "feature_names": ["Issue_Size", "Issue_Price", ...],
  "model_performance": {
    "Logistic Regression": {"auc": 0.7025, "accuracy": 0.656},
    "Random Forest": {"auc": 0.6837, "accuracy": 0.688},
    ...
  }
}
```

#### 3. Get Historical Data
```http
GET /historical-data
```

**Response:**
```json
{
  "count": 326,
  "data": [
    {
      "Date ": "03/02/10",
      "IPOName": "Infinite Comp",
      "Issue_Size": 189.8,
      "Subscription_QIB": 48.44,
      ...
    }
  ]
}
```

#### 4. Predict IPO Performance
```http
POST /predict
```

**Request Body:**
```json
{
  "ipo_name": "Example IPO Ltd",
  "issue_size": 1000.0,
  "issue_price": 250.0,
  "subscription_qib": 5.5,
  "subscription_hni": 3.2,
  "subscription_rii": 2.8,
  "subscription_total": 4.1
}
```

**Response:**
```json
{
  "ipo_name": "Example IPO Ltd",
  "prediction": "PROFITABLE",
  "probability": 0.78,
  "confidence": 78.0,
  "risk_level": "Low",
  "predicted_listing_gain_percent": 25.5,
  "predicted_opening_price": 312.75,
  "recommendation": "Strong BUY",
  "model_results": [
    {
      "model": "Ensemble",
      "probability": 0.75,
      "prediction": "PROFITABLE"
    },
    {
      "model": "Neural Network",
      "probability": 0.81,
      "prediction": "PROFITABLE"
    }
  ],
  "warning": null
}
```

### Error Responses

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `500`: Internal Server Error

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

### CORS Configuration

The API is configured to accept requests from any origin during development. For production, update the CORS settings in `backend/main.py` to restrict allowed origins.

### Mock Prediction Mode

If model files are not present in the `/model` directory, the API will operate in mock prediction mode, using simple heuristics based on subscription data. A warning will be included in the response.

## 📈 Results

### Model Performance

| Model | AUC Score | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.7025** | 0.656 | 0.64 | 0.63 | 0.635 |
| Random Forest | 0.6837 | 0.688 | 0.69 | 0.69 | 0.690 |
| SVM | 0.6562 | 0.641 | 0.65 | 0.64 | 0.645 |
| Voting Ensemble | 0.6611 | 0.672 | 0.67 | 0.67 | 0.670 |
| Neural Network | ~0.90* | 0.740 | 0.75 | 0.74 | 0.745 |

*Training AUC; validation performance varies

### Sample Prediction Output
```
IPO Prediction Results for MRF:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Prediction: PROFITABLE
💰 Expected Listing Gain: 31.24%
📈 Predicted Opening Price: ₹59.06
🎪 Neural Network Confidence: 64%
🗳️ Ensemble Confidence: 72%
💡 Recommendation: Moderate BUY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Key Insights
- 🏆 Logistic Regression surprisingly outperformed complex models
- 🎯 Optimal threshold identified at ~0.5 for balanced precision-recall
- 📊 Feature engineering significantly improved model performance
- 🎪 Ensemble methods provided more robust predictions

## 📁 Project Structure

```
INITIAL-IPO-PREDICTION/
├── 📊 Indian_IPO_Market_Data.csv    # Historical IPO dataset
├── 🐍 ML model working.py            # Original ML training script
├── 📖 README.md                      # This file
│
├── 🌐 frontend/                      # React Web Application
│   ├── public/                       # Static assets
│   ├── src/
│   │   ├── components/
│   │   │   ├── 3D/
│   │   │   │   └── Background3D.jsx  # 3D scene with Three.js
│   │   │   └── UI/
│   │   │       ├── PredictionForm.jsx
│   │   │       ├── PredictionResults.jsx
│   │   │       ├── VisualizationSection.jsx
│   │   │       ├── AboutSection.jsx
│   │   │       └── LoadingSpinner.jsx
│   │   ├── utils/
│   │   │   └── api.js               # API configuration
│   │   ├── App.jsx                  # Main app component
│   │   ├── App.css
│   │   ├── index.css                # Global styles with Tailwind
│   │   └── main.jsx                 # Entry point
│   ├── .env                         # Environment variables
│   ├── package.json                 # Dependencies
│   ├── vite.config.js               # Vite configuration
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   └── postcss.config.js            # PostCSS configuration
│
├── 🔧 backend/                       # FastAPI Backend
│   ├── main.py                      # FastAPI application
│   └── requirements.txt             # Python dependencies
│
└── 🤖 model/                         # ML Model Files Directory
    ├── README.md                    # Model setup instructions
    ├── ipo_scaler.pkl              # (to be placed here)
    ├── ipo_selected_features.pkl   # (to be placed here)
    ├── ipo_model_ensemble.pkl      # (to be placed here)
    └── ipo_model_neural_network.h5 # (to be placed here)
```

### Key Directories

- **`frontend/`**: React application with 3D UI built using Vite, React Three Fiber, and Tailwind CSS
- **`backend/`**: FastAPI server that loads ML models and provides prediction endpoints
- **`model/`**: Directory for trained model files (.pkl and .h5 files)

## 🎨 Screenshots

The web application features:
- **3D Landing Page**: Animated 3D background with floating geometric shapes and particle effects
- **Prediction Interface**: Intuitive form with glassmorphism design for inputting IPO details
- **Results Dashboard**: Animated cards displaying predictions, confidence scores, and recommendations
- **Analytics Section**: Interactive charts showing model performance and historical IPO data
- **About Section**: Detailed information about the ML models and methodology

## 🔒 Security & Best Practices

- ✅ Input validation on both frontend and backend
- ✅ CORS configuration for secure API access
- ✅ Environment variables for configuration
- ✅ Error handling throughout the application
- ✅ Mock prediction mode when models not available

## 🚀 Deployment

### Frontend Deployment
The frontend can be deployed to:
- **Vercel**: `vercel deploy`
- **Netlify**: `netlify deploy --prod`
- **GitHub Pages**: Build and deploy the `dist` folder

#### Deploy to Vercel (frontend only)
1. The repository root includes `vercel.json`, so Vercel will automatically run `cd frontend && npm install && npm run build` and publish `frontend/dist`.
2. In the Vercel dashboard, add an environment variable `VITE_API_URL` that points to your deployed backend (e.g., `https://your-backend.example.com`).
3. Import the GitHub repository into Vercel and select **Deploy**. For local/CLI use, run `vercel --prod` from the repo root.
4. Client-side routes are pre-configured to fall back to `index.html`, so deep links work without extra rewrites.

### Backend Deployment
The backend can be deployed to:
- **Heroku**: Using Procfile with uvicorn
- **AWS EC2**: Running uvicorn as a service
- **Docker**: Containerize the backend application

## 📝 Notes

- The backend replicates the **exact feature engineering** from `ML model working.py` to ensure model compatibility
- Model files must be generated by running the training script first
- The application works in mock mode if model files are not present (with warnings)
- All predictions are for informational purposes only - always do your own research before investing

## 📁 Project Structure (Original)

```
ipo-prediction/
├── 📊 data/
│   ├── raw/
│   │   └── ipo_data.csv
│   └── processed/
│       └── preprocessed_data.pkl
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── 🐍 src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
├── 🤖 models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── neural_network.h5
│   └── ensemble_model.pkl
├── 📊 results/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── classification_reports/
├── 📋 requirements.txt
├── 🔧 config.yaml
└── 📖 README.md
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 Reporting Bugs
1. Check existing issues first
2. Create detailed bug reports with:
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error logs

### ✨ Suggesting Features
1. Open an issue with the "enhancement" label
2. Describe the feature and its benefits
3. Provide use cases and examples

### 🔧 Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/ipo-prediction.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create pull request
git push origin feature/amazing-feature
```

### 📝 Pull Request Guidelines
- Follow the existing code style
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Pratyush Rawat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software")...
```

## 📞 Contact

### 👨‍💻 Author
**Pratyush Rawat**



<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for the Machine Learning community

</div>![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/FLACK277/INITIAL-IPO-PREDICTION?utm_source=oss&utm_medium=github&utm_campaign=FLACK277%2FINITIAL-IPO-PREDICTION&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
