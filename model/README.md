# Model Files Directory

This directory should contain the trained IPO prediction model files.

## Required Files

Place the following files in this directory for the backend to work with trained models:

1. **ipo_scaler.pkl** - StandardScaler for feature normalization
2. **ipo_selected_features.pkl** - List of selected features for the model
3. **ipo_model_ensemble.pkl** - Trained voting ensemble model (Random Forest, Gradient Boosting, Logistic Regression, SVM)
4. **ipo_model_neural_network.h5** - Trained neural network model

## Generating Model Files

To generate these model files, run the training script from the root directory:

```bash
python "ML model working.py"
```

This will train the models and save the required `.pkl` and `.h5` files to the current directory. Then move them to this `/model` directory:

```bash
mv ipo_scaler.pkl model/
mv ipo_selected_features.pkl model/
mv ipo_model_ensemble.pkl model/
mv ipo_model_neural_network.h5 model/
```

## Fallback Mode

If these files are not present, the backend API will operate in **mock prediction mode**, using simple heuristics based on subscription data to provide example predictions. A warning will be included in the API response.

## File Descriptions

### ipo_scaler.pkl
- **Type**: joblib serialized StandardScaler
- **Purpose**: Scales input features to normalize them before feeding to models
- **Created by**: Training pipeline after fitting on training data

### ipo_selected_features.pkl
- **Type**: joblib serialized list/array
- **Purpose**: Defines which engineered features the models were trained on
- **Content**: List of feature names (e.g., 'Issue_Size', 'Log_Subscription_QIB', 'QIB_RII_Interaction', etc.)

### ipo_model_ensemble.pkl
- **Type**: joblib serialized VotingClassifier
- **Purpose**: Ensemble model combining multiple classifiers
- **Components**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Voting**: Soft voting (uses predicted probabilities)

### ipo_model_neural_network.h5
- **Type**: Keras/TensorFlow saved model
- **Purpose**: Deep learning model for IPO prediction
- **Architecture**: 2 hidden layers (384, 128 units) with dropout and L2 regularization
- **Performance**: ~90% training AUC, 74% validation accuracy
