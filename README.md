# 🚀 IPO Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Scikit--learn-orange)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)

*Predicting IPO profitability using advanced machine learning techniques and ensemble methods*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-machine-learning-models) • [Results](#-results) • [Contributing](#-contributing)

</div>

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
pip or conda package manager
```

### Clone Repository
```bash
git clone https://github.com/yourusername/ipo-prediction.git
cd ipo-prediction
```

### Install Dependencies
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

</div>
