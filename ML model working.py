import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.combine import SMOTETomek
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam, AdamW
from tensorflow.keras.metrics import Precision, Recall, AUC
import keras_tuner as kt
from keras_tuner.tuners import Hyperband, BayesianOptimization
import joblib
import warnings
import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess data
def load_and_preprocess_data(filepath):
    """Load and preprocess IPO data"""
    df = pd.read_csv('Downloads/Indian_IPO_Market_Data.csv')
    
    # Store original IPO names for later use in predictions
    ipo_names = df['IPOName'].copy() if 'IPOName' in df.columns else None
    
    # Basic preprocessing
    df['Listing_Gains_Profit'] = np.where(df['Listing_Gains_Percent'] > 0, 1, 0)
    
    # Keep IPO name separate for reference but remove from training data
    if 'IPOName' in df.columns:
        ipo_name_df = df[['IPOName']].copy()
        df = df.drop(['Date ', 'IPOName'], axis=1)
    else:
        ipo_name_df = None
        df = df.drop(['Date '], axis=1)
    
    # Save the percentage for later analysis
    listing_gains_percent = df['Listing_Gains_Percent'].copy()
    df = df.drop(['Listing_Gains_Percent'], axis=1)
    
    # Advanced feature engineering
    df = create_features(df)
    
    return df, ipo_name_df, listing_gains_percent

def create_features(df):
    """Create additional features through feature engineering"""
    # Interaction features
    df['QIB_RII_Interaction'] = df['Subscription_QIB'] * df['Subscription_RII']
    df['HNI_RII_Interaction'] = df['Subscription_HNI'] * df['Subscription_RII']
    df['QIB_HNI_Interaction'] = df['Subscription_QIB'] * df['Subscription_HNI']
    df['Size_Price_Ratio'] = df['Issue_Size'] / (df['Issue_Price'] + 1e-8)
    df['Subscription_Imbalance'] = (df['Subscription_QIB'] - df['Subscription_RII']).abs()
    df['Price_QIB_Interaction'] = df['Issue_Price'] * df['Subscription_QIB']
    df['Size_QIB_Interaction'] = df['Issue_Size'] * df['Subscription_QIB']
    
    # Logarithmic transformations for highly skewed features
    for col in ['Issue_Size', 'Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Subscription_Total']:
        df[f'Log_{col}'] = np.log1p(df[col])
    
    # Polynomial features
    df['Issue_Size_Squared'] = df['Issue_Size'] ** 2
    df['Issue_Price_Squared'] = df['Issue_Price'] ** 2
    
    # Statistical features
    df['Subscription_Mean'] = df[['Subscription_QIB', 'Subscription_HNI', 'Subscription_RII']].mean(axis=1)
    df['Subscription_Std'] = df[['Subscription_QIB', 'Subscription_HNI', 'Subscription_RII']].std(axis=1)
    df['Subscription_Range'] = df[['Subscription_QIB', 'Subscription_HNI', 'Subscription_RII']].max(axis=1) - df[['Subscription_QIB', 'Subscription_HNI', 'Subscription_RII']].min(axis=1)
    df['QIB_to_RII_Ratio'] = df['Subscription_QIB'] / (df['Subscription_RII'] + 1e-8)
    df['HNI_to_RII_Ratio'] = df['Subscription_HNI'] / (df['Subscription_RII'] + 1e-8)
    
    # Composite features
    df['Weighted_Subscription'] = (df['Subscription_QIB'] * 0.5 + 
                                 df['Subscription_HNI'] * 0.3 + 
                                 df['Subscription_RII'] * 0.2)
    
    return df

def handle_outliers(df, columns, method='iqr'):
    """Handle outliers using chosen method"""
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in columns:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    elif method == 'percentile':
        for col in columns:
            lower_bound = df_clean[col].quantile(0.01)
            upper_bound = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    elif method == 'zscore':
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean[col] = df_clean[col].clip(mean - 3 * std, mean + 3 * std)
            
    return df_clean

def visualize_data(df, listing_gains_percent=None):
    """Create comprehensive visualizations of the data"""
    plt.figure(figsize=(20, 15))
    
    # Target distribution
    plt.subplot(3, 3, 1)
    sns.countplot(data=df, x='Listing_Gains_Profit')
    plt.title('Target Variable Distribution')
    
    # Numerical features distribution
    plt.subplot(3, 3, 2)
    sns.histplot(data=df, x='Issue_Size', kde=True)
    plt.title('Issue Size Distribution')
    
    plt.subplot(3, 3, 3)
    sns.histplot(data=df, x='Issue_Price', kde=True)
    plt.title('Issue Price Distribution')
    
    # Subscription relationships
    plt.subplot(3, 3, 4)
    sns.scatterplot(data=df, x='Subscription_RII', y='Subscription_Total', hue='Listing_Gains_Profit')
    plt.title('Retail vs Total Subscription')
    
    # Correlation matrix
    plt.subplot(3, 3, 5)
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Limit to important features for clarity
    important_cols = ['Issue_Size', 'Issue_Price', 'Subscription_QIB', 'Subscription_HNI', 
                     'Subscription_RII', 'Subscription_Total', 'Listing_Gains_Profit']
    important_cols = [col for col in important_cols if col in numeric_cols]
    corr = df[important_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    
    # Boxplots
    plt.subplot(3, 3, 6)
    sns.boxplot(data=df, x='Listing_Gains_Profit', y='Issue_Price')
    plt.title('Issue Price by Profitability')
    
    # Distribution of listing gains percentage if available
    if listing_gains_percent is not None:
        plt.subplot(3, 3, 7)
        sns.histplot(listing_gains_percent, kde=True)
        plt.title('Distribution of Listing Gains Percentage')
        plt.axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance visualization
    plt.figure(figsize=(12, 8))
    X = df.drop('Listing_Gains_Profit', axis=1)
    y = df['Listing_Gains_Profit']
    
    # Use Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Plot feature importances
    features = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15 features
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def prepare_data(df, test_size=0.2):
    """Prepare data for modeling with advanced techniques"""
    # Separate features and target
    X = df.drop('Listing_Gains_Profit', axis=1)
    y = df['Listing_Gains_Profit']
    
    # Feature selection using multiple methods
    # 1. Statistical tests (ANOVA)
    selector1 = SelectKBest(f_classif, k=15)
    X_selected1 = selector1.fit_transform(X, y)
    selected_features1 = X.columns[selector1.get_support()]
    
    # 2. Mutual information
    selector2 = SelectKBest(mutual_info_classif, k=15)
    X_selected2 = selector2.fit_transform(X, y)
    selected_features2 = X.columns[selector2.get_support()]
    
    # 3. Recursive Feature Elimination with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector3 = RFE(rf, n_features_to_select=15)
    X_selected3 = selector3.fit_transform(X, y)
    selected_features3 = X.columns[selector3.support_]
    
    # Combine selected features from different methods
    all_selected_features = list(set(list(selected_features1) + list(selected_features2) + list(selected_features3)))
    X = X[all_selected_features]
    
    print(f"Selected {len(all_selected_features)} features: {all_selected_features}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,
        random_state=42
    )
    
    # Advanced scaling with PowerTransformer for handling skewed features
    scaler = PowerTransformer(method='yeo-johnson')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Handle class imbalance with SMOTETomek (combines over and under sampling)
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train_scaled, y_train)
    
    # Save the scaler and selected features
    joblib.dump(scaler, 'ipo_scaler.pkl')
    joblib.dump(all_selected_features, 'ipo_selected_features.pkl')
    
    return X_train_res, X_test_scaled, y_train_res, y_test, all_selected_features, scaler

def build_model(hp):
    """Build a tunable neural network model with reduced search space"""
    # Get input shape from hp object
    try:
        input_shape = hp.values['input_shape']
    except (KeyError, AttributeError):
        # Default fallback if not available
        input_shape = 15  # Typical number of selected features
    
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(input_shape,)))
    
    # Add an initial normalization layer
    model.add(layers.BatchNormalization())
    
    # Fixed to 2 hidden layers to reduce search space
    num_layers = 2
    
    # First hidden layer with more units, but reduced options
    units = hp.Int('units_0', 128, 384, step=128)  # Reduced options: 128, 256, 384
    activation = hp.Choice('activation_0', ['relu', 'swish'])  # Reduced options
    dropout_rate = hp.Float('dropout_0', 0.2, 0.4, step=0.1)  # Discretized
    
    model.add(layers.Dense(
        units=units,
        activation=activation,
        kernel_regularizer=regularizers.l2(hp.Choice('l2_0', [1e-5, 1e-4, 1e-3]))  # Discrete options
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Second hidden layer with fewer options
    units = hp.Int('units_1', 64, 192, step=64)  # Reduced options: 64, 128, 192
    activation = hp.Choice('activation_1', ['relu', 'swish'])  # Reduced options
    dropout_rate = hp.Float('dropout_1', 0.2, 0.4, step=0.1)  # Discretized
    
    model.add(layers.Dense(
        units=units,
        activation=activation,
        kernel_regularizer=regularizers.l2(hp.Choice('l2_1', [1e-5, 1e-4, 1e-3]))  # Discrete options
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Reduced optimizer options
    optimizer_choice = hp.Choice('optimizer', ['adam', 'adamw'])
    lr = hp.Choice('lr', [1e-4, 5e-4, 1e-3, 3e-3])  # Discrete learning rates
    
    if optimizer_choice == 'adam':
        opt = Adam(learning_rate=lr)
    else:
        opt = AdamW(learning_rate=lr, weight_decay=hp.Choice('weight_decay', [1e-5, 1e-4]))
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape):
    """Train model with hyperparameter tuning - limited to around 200 trials"""
    # Define a custom build_model function that captures the input_shape
    def build_model_with_shape(hp):
        return build_model(hp)
    
    # Use BayesianOptimization instead of Hyperband with fixed max_trials
    tuner = BayesianOptimization(
        build_model_with_shape,
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=200,  # Explicitly limit to 200 trials
        directory='keras_tuner',
        project_name='ipo_prediction_limited',
        overwrite=True
    )
    
    # Register the fixed hyperparameter outside the tuner creation
    hp = tuner.oracle.hyperparameters
    hp.Fixed('input_shape', value=input_shape)
    
    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=20,  # Reduced from 30
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=10,  # Reduced from 15
        min_lr=1e-6
    )
    
    checkpoint = ModelCheckpoint(
        'best_model_checkpoint.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
    
    # Calculate class weights for imbalanced data
    class_weights = {0: 1., 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
    
    print("Starting hyperparameter search with limit of 200 trials...")
    tuner.search(
        X_train, y_train,
        epochs=150,  # Reduced from 250
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, checkpoint],
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )
    
    # Get the best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    
    # Fine-tune the best model with more epochs
    print("\nFine-tuning the best model:")
    history = best_model.fit(
        X_train, y_train,
        epochs=250,  # Reduced from 350
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, checkpoint],
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return best_model, best_hps

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model performance with adjustable threshold"""
    y_proba = model.predict(X_test)
    y_pred = (y_proba > threshold).astype(int)
    
    # Classification report
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y_test, y_pred))
    
    # Calculate key metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Profitable', 'Profitable'], 
                yticklabels=['Not Profitable', 'Profitable'])
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.show()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    precision_values = []
    recall_values = []
    f1_values = []
    accuracy_values = []
    thresholds_to_try = np.linspace(0.1, 0.9, 17)  # More granular threshold testing
    
    for t in thresholds_to_try:
        y_pred_t = (y_proba > t).astype(int)
        precision_values.append(precision_score(y_test, y_pred_t))
        recall_values.append(recall_score(y_test, y_pred_t))
        f1_values.append(f1_score(y_test, y_pred_t))
        accuracy_values.append(accuracy_score(y_test, y_pred_t))
    
    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_to_try, precision_values, 'b-', label='Precision')
    plt.plot(thresholds_to_try, recall_values, 'g-', label='Recall')
    plt.plot(thresholds_to_try, f1_values, 'r-', label='F1 Score')
    plt.plot(thresholds_to_try, accuracy_values, 'y-', label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find threshold with best F1 score
    best_f1_idx = np.argmax(f1_values)
    best_f1_threshold = thresholds_to_try[best_f1_idx]
    print(f"Best threshold for F1: {best_f1_threshold:.2f} (F1: {f1_values[best_f1_idx]:.4f})")
    
    # Find threshold with best accuracy
    best_acc_idx = np.argmax(accuracy_values)
    best_acc_threshold = thresholds_to_try[best_acc_idx]
    print(f"Best threshold for Accuracy: {best_acc_threshold:.2f} (Accuracy: {accuracy_values[best_acc_idx]:.4f})")
    
    # Evaluate with best F1 threshold
    y_pred_best = (y_proba > best_f1_threshold).astype(int)
    print(f"\nClassification Report with best F1 threshold ({best_f1_threshold:.2f}):")
    print(classification_report(y_test, y_pred_best))
    
    return best_f1_threshold
def train_ensemble_models(X_train, y_train, X_test, y_test):
    """Train and evaluate ensemble models for IPO prediction"""
    print("\n===== Training Ensemble Models =====")
    
    # Train individual models
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        random_state=42
    )
    lr.fit(X_train, y_train)
    
    # SVM
    print("Training SVM...")
    svm = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    svm.fit(X_train, y_train)
    
    # Create a voting classifier
    print("Creating Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr),
            ('svm', svm)
        ],
        voting='soft'  # Use probability predictions
    )
    voting_clf.fit(X_train, y_train)
    
    # Evaluate individual models
    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Logistic Regression': lr,
        'SVM': svm,
        'Voting Ensemble': voting_clf
    }
    
    print("\n===== Model Evaluation =====")
    best_auc = 0
    best_model = None
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n{name} results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
    
    print(f"\nBest traditional model based on AUC: {list(models.keys())[list(models.values()).index(best_model)]}")
    
    # Plot ROC curves for all models
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return voting_clf  # Return the ensemble model

def make_prediction(models, scaler, input_data, features, threshold=0.5):
    """Make prediction on new data using multiple models"""
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data], columns=features)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    results = []
    
    # Make predictions with each model
    for model_name, model in models.named_estimators_.items():
        # Neural network models
        if isinstance(model, keras.Model):
            prediction_prob = model.predict(input_scaled)[0][0]
        # Traditional ML models
        else:
            prediction_prob = model.predict_proba(input_scaled)[0][1]
            
        prediction = 1 if prediction_prob > threshold else 0
        profit_status = "PROFITABLE" if prediction == 1 else "NOT PROFITABLE"
        
        results.append({
            'model': model_name,
            'probability': prediction_prob,
            'prediction': prediction,
            'status': profit_status
        })
    
    return results

def predict_ipo(models, scaler, selected_features, ipo_data):
    """Make prediction for a specific IPO"""
    # Prepare the IPO data
    input_data = {}
    
    # Extract the base features that are required
    required_features = ['Issue_Size', 'Issue_Price', 'Subscription_QIB', 
                         'Subscription_HNI', 'Subscription_RII', 'Subscription_Total']
    
    for feature in required_features:
        if feature in ipo_data:
            input_data[feature] = ipo_data[feature]
        else:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Create engineered features
    engineered_data = input_data.copy()
    
    # Interaction features
    engineered_data['QIB_RII_Interaction'] = input_data['Subscription_QIB'] * input_data['Subscription_RII']
    engineered_data['HNI_RII_Interaction'] = input_data['Subscription_HNI'] * input_data['Subscription_RII']
    engineered_data['QIB_HNI_Interaction'] = input_data['Subscription_QIB'] * input_data['Subscription_HNI']
    engineered_data['Size_Price_Ratio'] = input_data['Issue_Size'] / (input_data['Issue_Price'] + 1e-8)
    engineered_data['Subscription_Imbalance'] = abs(input_data['Subscription_QIB'] - input_data['Subscription_RII'])
    engineered_data['Price_QIB_Interaction'] = input_data['Issue_Price'] * input_data['Subscription_QIB']
    engineered_data['Size_QIB_Interaction'] = input_data['Issue_Size'] * input_data['Subscription_QIB']
    
    # Logarithmic transformations
    for col in ['Issue_Size', 'Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Subscription_Total']:
        engineered_data[f'Log_{col}'] = np.log1p(input_data[col])
    
    # Polynomial features
    engineered_data['Issue_Size_Squared'] = input_data['Issue_Size'] ** 2
    engineered_data['Issue_Price_Squared'] = input_data['Issue_Price'] ** 2
    
    # Statistical features
    engineered_data['Subscription_Mean'] = np.mean([input_data['Subscription_QIB'], 
                                                  input_data['Subscription_HNI'], 
                                                  input_data['Subscription_RII']])
    engineered_data['Subscription_Std'] = np.std([input_data['Subscription_QIB'], 
                                                input_data['Subscription_HNI'], 
                                                input_data['Subscription_RII']])
    engineered_data['Subscription_Range'] = max([input_data['Subscription_QIB'], 
                                               input_data['Subscription_HNI'], 
                                               input_data['Subscription_RII']]) - min([input_data['Subscription_QIB'], 
                                                                                    input_data['Subscription_HNI'], 
                                                                                    input_data['Subscription_RII']])
    engineered_data['QIB_to_RII_Ratio'] = input_data['Subscription_QIB'] / (input_data['Subscription_RII'] + 1e-8)
    engineered_data['HNI_to_RII_Ratio'] = input_data['Subscription_HNI'] / (input_data['Subscription_RII'] + 1e-8)
    
    # Composite features
    engineered_data['Weighted_Subscription'] = (input_data['Subscription_QIB'] * 0.5 + 
                                             input_data['Subscription_HNI'] * 0.3 + 
                                             input_data['Subscription_RII'] * 0.2)
    
    # Filter to only include selected features
    filtered_data = {feature: engineered_data[feature] for feature in selected_features if feature in engineered_data}
    
    # Make sure all selected features are present
    for feature in selected_features:
        if feature not in filtered_data:
            filtered_data[feature] = 0  # Default to 0 for missing features
    
    # Make prediction using the ensemble of models
    results = make_prediction(models, scaler, filtered_data, selected_features)
    
    return results

def save_models(models):
    """Save all trained models to disk"""
    for name, model in models.items():
        if isinstance(model, keras.Model):
            model.save(f'ipo_model_{name}.h5')
        else:
            joblib.dump(model, f'ipo_model_{name}.pkl')
    print("All models saved successfully.")

def load_models(model_paths):
    """Load trained models from disk"""
    models = {}
    for name, path in model_paths.items():
        if path.endswith('.h5'):
            models[name] = keras.models.load_model(path)
        else:
            models[name] = joblib.load(path)
    return models
def get_user_input():
    """Get IPO details from user"""
    print("\n===== Enter IPO Details =====")
    
    ipo_data = {}
    ipo_data['IPOName'] = input("IPO Name: ")
    
    try:
        ipo_data['Issue_Size'] = float(input("Issue Size (in crores): "))
        ipo_data['Issue_Price'] = float(input("Issue Price (in INR): "))
        ipo_data['Subscription_QIB'] = float(input("QIB Subscription (times): "))
        ipo_data['Subscription_HNI'] = float(input("HNI Subscription (times): "))
        ipo_data['Subscription_RII'] = float(input("Retail Subscription (times): "))
        ipo_data['Subscription_Total'] = float(input("Total Subscription (times): "))
    except ValueError:
        print("Error: Please enter numeric values for all fields except IPO Name.")
        return None
    
    return ipo_data
def predict_opening_price(issue_price, subscription_metrics, avg_probability, consensus):
    """
    Predict the opening price of an IPO based on issue price and model predictions
    
    Parameters:
    -----------
    issue_price : float
        The issue price of the IPO
    subscription_metrics : dict
        Dictionary containing subscription details (QIB, HNI, RII, Total)
    avg_probability : float
        The average probability from model predictions (0-1)
    consensus : str
        The consensus prediction ("PROFITABLE" or "NOT PROFITABLE")
        
    Returns:
    --------
    float
        Predicted opening price
    float
        Estimated listing gain/loss percentage
    """
    # Base parameters for gain calculation
    base_gain = 10  # Minimum expected gain for profitable IPOs
    max_gain = 50   # Maximum expected gain for highly subscribed IPOs
    base_loss = -5  # Minimum expected loss for unprofitable IPOs
    max_loss = -20  # Maximum expected loss for poorly subscribed IPOs
    
    # Calculate subscription strength score (0-1)
    qib_weight, hni_weight, rii_weight = 0.5, 0.3, 0.2  # Weights for different investor categories
    
    # Normalize subscription values using log scale to handle extreme values
    qib_norm = min(1.0, np.log1p(subscription_metrics['Subscription_QIB']) / np.log1p(100))
    hni_norm = min(1.0, np.log1p(subscription_metrics['Subscription_HNI']) / np.log1p(100))
    rii_norm = min(1.0, np.log1p(subscription_metrics['Subscription_RII']) / np.log1p(50))
    total_norm = min(1.0, np.log1p(subscription_metrics['Subscription_Total']) / np.log1p(100))
    
    # Weighted subscription score
    subscription_score = (qib_norm * qib_weight + 
                         hni_norm * hni_weight + 
                         rii_norm * rii_weight)
    
    # Boost subscription score based on total subscription
    subscription_score = 0.7 * subscription_score + 0.3 * total_norm
    
    # Calculate expected gain/loss percentage
    if consensus == "PROFITABLE":
        # For profitable predictions, scale between base_gain and max_gain
        # Use both model confidence and subscription metrics
        confidence_factor = (avg_probability - 0.5) * 2  # Map from [0.5, 1.0] to [0, 1.0]
        
        # Combine confidence and subscription score with non-linear scaling
        combined_score = 0.6 * confidence_factor + 0.4 * subscription_score
        combined_score = combined_score ** 0.8  # Non-linear scaling to favor higher values
        
        listing_gain_percent = base_gain + (max_gain - base_gain) * combined_score
        
        # Apply a small correction based on issue price (higher priced IPOs tend to have lower % gains)
        price_dampening = max(0.8, 1 - (issue_price / 1000) * 0.1)
        listing_gain_percent *= price_dampening
    else:
        # For unprofitable predictions
        confidence_factor = (0.5 - avg_probability) * 2  # Map from [0, 0.5] to [1.0, 0]
        combined_score = 0.7 * confidence_factor + 0.3 * (1 - subscription_score)
        listing_gain_percent = base_loss - (max_loss - base_loss) * combined_score
    
    # Calculate predicted opening price
    predicted_opening_price = issue_price * (1 + listing_gain_percent / 100)
    
    # Round to 2 decimal places
    predicted_opening_price = round(predicted_opening_price, 2)
    listing_gain_percent = round(listing_gain_percent, 2)
    
    return predicted_opening_price, listing_gain_percent

def predict_ipo_interactive():
    """Interactive function to predict IPO listing gains"""
    print("===== IPO Listing Gain Prediction System =====")
    print("📈 Enter the following details about the IPO:")
    
    try:
        ipo_name = input("Enter IPO Name: ")
        issue_size = float(input("Enter Issue Size (in crores): "))
        issue_price = float(input("Enter Issue Price (in INR): "))
        subscription_qib = float(input("Enter QIB Subscription (times): "))
        subscription_hni = float(input("Enter HNI Subscription (times): "))
        subscription_rii = float(input("Enter Retail/RII Subscription (times): "))
        subscription_total = float(input("Enter Total Subscription (times): "))
    except ValueError:
        print("❌ Invalid input. Please enter numerical values for all fields except IPO Name.")
        return
    
    # Create input data dictionary
    ipo_data = {
        'IPOName': ipo_name,
        'Issue_Size': issue_size,
        'Issue_Price': issue_price,
        'Subscription_QIB': subscription_qib,
        'Subscription_HNI': subscription_hni,
        'Subscription_RII': subscription_rii,
        'Subscription_Total': subscription_total
    }
    
    # Load the saved models and scaler
    try:
        scaler = joblib.load('ipo_scaler.pkl')
        selected_features = joblib.load('ipo_selected_features.pkl')
        
        models = {}
        # Load neural network model
        if os.path.exists('ipo_model_neural_network.h5'):
            nn_model = keras.models.load_model('ipo_model_neural_network.h5')
            dummy_input = np.zeros((1, nn_model.input_shape[1]))
            _ = nn_model.predict(dummy_input, verbose=0)
            models['Neural Network'] = nn_model
        
        # Load ensemble model
        if os.path.exists('ipo_model_ensemble.pkl'):
            models['Ensemble'] = joblib.load('ipo_model_ensemble.pkl')
            
        if not models:
            print("⚠️ No trained models found. Please run the training process first.")
            return
            
    except FileNotFoundError:
        print("⚠️ Model files not found. Please run the training process first.")
        return
    
    # Generate engineered features
    engineered_data = ipo_data.copy()
    
    # Interaction features
    engineered_data['QIB_RII_Interaction'] = ipo_data['Subscription_QIB'] * ipo_data['Subscription_RII']
    engineered_data['HNI_RII_Interaction'] = ipo_data['Subscription_HNI'] * ipo_data['Subscription_RII']
    engineered_data['QIB_HNI_Interaction'] = ipo_data['Subscription_QIB'] * ipo_data['Subscription_HNI']
    engineered_data['Size_Price_Ratio'] = ipo_data['Issue_Size'] / (ipo_data['Issue_Price'] + 1e-8)
    engineered_data['Subscription_Imbalance'] = abs(ipo_data['Subscription_QIB'] - ipo_data['Subscription_RII'])
    engineered_data['Price_QIB_Interaction'] = ipo_data['Issue_Price'] * ipo_data['Subscription_QIB']
    engineered_data['Size_QIB_Interaction'] = ipo_data['Issue_Size'] * ipo_data['Subscription_QIB']
    
    # Logarithmic transformations
    for col in ['Issue_Size', 'Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Subscription_Total']:
        engineered_data[f'Log_{col}'] = np.log1p(ipo_data[col])
    
    # Polynomial features
    engineered_data['Issue_Size_Squared'] = ipo_data['Issue_Size'] ** 2
    engineered_data['Issue_Price_Squared'] = ipo_data['Issue_Price'] ** 2
    
    # Statistical features
    engineered_data['Subscription_Mean'] = np.mean([ipo_data['Subscription_QIB'], 
                                                  ipo_data['Subscription_HNI'], 
                                                  ipo_data['Subscription_RII']])
    engineered_data['Subscription_Std'] = np.std([ipo_data['Subscription_QIB'], 
                                                ipo_data['Subscription_HNI'], 
                                                ipo_data['Subscription_RII']])
    engineered_data['Subscription_Range'] = max([ipo_data['Subscription_QIB'], 
                                               ipo_data['Subscription_HNI'], 
                                               ipo_data['Subscription_RII']]) - min([ipo_data['Subscription_QIB'], 
                                                                                    ipo_data['Subscription_HNI'], 
                                                                                    ipo_data['Subscription_RII']])
    engineered_data['QIB_to_RII_Ratio'] = ipo_data['Subscription_QIB'] / (ipo_data['Subscription_RII'] + 1e-8)
    engineered_data['HNI_to_RII_Ratio'] = ipo_data['Subscription_HNI'] / (ipo_data['Subscription_RII'] + 1e-8)
    
    # Composite features
    engineered_data['Weighted_Subscription'] = (ipo_data['Subscription_QIB'] * 0.5 + 
                                             ipo_data['Subscription_HNI'] * 0.3 + 
                                             ipo_data['Subscription_RII'] * 0.2)
    
    # Filter to only include selected features
    filtered_data = {feature: engineered_data[feature] for feature in selected_features if feature in engineered_data}
    
    # Make sure all selected features are present
    missing_features = []
    for feature in selected_features:
        if feature not in filtered_data:
            filtered_data[feature] = 0  # Default to 0 for missing features
            missing_features.append(feature)
    
    if missing_features:
        print(f"⚠️ Note: The following features were not available and defaulted to 0: {', '.join(missing_features)}")
    
    # Make predictions
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([filtered_data], columns=selected_features)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    results = []
    
    # Make predictions with each model
    for model_name, model in models.items():
        # Neural network models
        if isinstance(model, keras.Model):
            prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
        # Traditional ML models
        else:
            prediction_prob = model.predict_proba(input_scaled)[0][1]
            
        prediction = 1 if prediction_prob > 0.5 else 0
        profit_status = "PROFITABLE" if prediction == 1 else "NOT PROFITABLE"
        
        results.append({
            'model': model_name,
            'probability': prediction_prob,
            'prediction': prediction,
            'status': profit_status
        })
    
    # Calculate consensus prediction
    avg_probability = sum(result['probability'] for result in results) / len(results)
    consensus = "PROFITABLE" if avg_probability > 0.5 else "NOT PROFITABLE"
    confidence = max(avg_probability, 1 - avg_probability) * 100

    # Predict opening price and expected listing gain/loss
    predicted_opening_price, listing_gain_percent = predict_opening_price(
        issue_price, 
        ipo_data, 
        avg_probability, 
        consensus
    )
    
    # Display results
    print("\n📊 IPO Prediction Results:")
    print(f"IPO Name: {ipo_data['IPOName']}")
    print(f"Issue Size: ₹{ipo_data['Issue_Size']} Cr")
    print(f"Issue Price: ₹{ipo_data['Issue_Price']}")
    print(f"Subscription Details:")
    print(f"  - QIB: {ipo_data['Subscription_QIB']}x")
    print(f"  - HNI: {ipo_data['Subscription_HNI']}x")
    print(f"  - RII: {ipo_data['Subscription_RII']}x")
    print(f"  - Total: {ipo_data['Subscription_Total']}x")
    
    print("\n🔮 Model Predictions:")
    for result in results:
        emoji = "✅" if result['status'] == "PROFITABLE" else "❌"
        print(f"{emoji} {result['model']}: {result['status']} (Probability: {result['probability']:.2f})")
    
    # Consensus emoji
    consensus_emoji = "✅" if consensus == "PROFITABLE" else "❌"
    print(f"\n🔍 Consensus: {consensus_emoji} {consensus} (Confidence: {confidence:.2f}%)")
    
    # Display expected listing gain/loss and predicted opening price
    gain_emoji = "📈" if listing_gain_percent > 0 else "📉"
    print(f"{gain_emoji} Expected Listing {'Gain' if listing_gain_percent > 0 else 'Loss'}: {listing_gain_percent:.2f}%")
    print(f"💰 Predicted Opening Price: ₹{predicted_opening_price:.2f}")

    # Calculate absolute profit/loss per share
    absolute_change = predicted_opening_price - ipo_data['Issue_Price']
    if absolute_change > 0:
        print(f"💵 Expected Profit per Share: ₹{absolute_change:.2f}")
    else:
        print(f"💸 Expected Loss per Share: ₹{abs(absolute_change):.2f}")

    # Estimate listing gain
    if consensus == "PROFITABLE":
        # Simple regression model to convert confidence into estimated gain percentage
        # These parameters should ideally be trained on historical data
        base_gain = 10  # Minimum expected gain for profitable IPOs
        max_gain = 40   # Maximum expected gain
        
        # Map probability from [0.5, 1.0] to [base_gain, max_gain] using a sigmoid-like curve
        normalized_prob = (avg_probability - 0.5) * 2  # Maps [0.5, 1.0] to [0, 1.0]
        estimated_gain = base_gain + (max_gain - base_gain) * (normalized_prob ** 0.7)  # Non-linear scaling
        
        # Adjust based on subscription metrics
        subscription_factor = min(1.5, (ipo_data['Subscription_Total'] / 50))  # Cap at 1.5x boost
        expected_gain = estimated_gain * subscription_factor
        
        print(f"📈 Expected Listing Gain: {expected_gain:.2f}%")
    else:
        # For non-profitable predictions, estimate negative gains
        normalized_prob = avg_probability * 2  # Maps [0, 0.5] to [0, 1.0]
        expected_loss = -20 * (1 - normalized_prob ** 0.7)  # Max negative gain of -20%
        print(f"📉 Expected Listing Loss: {expected_loss:.2f}%")
    
    # Provide investment recommendation
    print("\n💡 Investment Recommendation:")
    if consensus == "PROFITABLE" and confidence > 70:
        print("Strong BUY recommendation. High confidence in positive listing gains.")
    elif consensus == "PROFITABLE" and confidence > 60:
        print("Moderate BUY recommendation. Consider investing based on your risk appetite.")
    elif consensus == "PROFITABLE":
        print("Cautious BUY recommendation. There's potential upside but with moderate risk.")
    elif consensus == "NOT PROFITABLE" and confidence > 70:
        print("Strong AVOID recommendation. High confidence in negative listing performance.")
    else:
        print("AVOID recommendation. Better investment opportunities may be available.")
    
    print("\n===== Analysis Complete =====")
    
    # Ask if user wants to analyze another IPO
    another = input("\nWould you like to analyze another IPO? (yes/no): ").lower()
    if another in ['yes', 'y']:
        predict_ipo_interactive()
def main():
    """Main function to run the IPO prediction system"""
    print("===== IPO Listing Gain Prediction System =====")
    # Check if models are already trained and saved
    try:
        # Try to load saved models and components
        print("Loading saved models and components...")
        model = joblib.load('ipo_model_ensemble.pkl')
        scaler = joblib.load('ipo_scaler.pkl')
        selected_features = joblib.load('ipo_selected_features.pkl')
        print("Models loaded successfully!")
        
        # Get IPO details from user
        while True:
            ipo_data = get_user_input()
            
            if ipo_data is None:
                continue
                
            # Make prediction
            result = predict_ipo(model, scaler, selected_features, ipo_data)
            
            # Calculate consensus prediction
            avg_probability = sum(r['probability'] for r in result) / len(result)
            consensus = "PROFITABLE" if avg_probability > 0.5 else "NOT PROFITABLE"
            confidence = max(avg_probability, 1 - avg_probability) * 100
            
            # Predict opening price
            predicted_opening_price, listing_gain_percent = predict_opening_price(
                ipo_data['Issue_Price'], 
                ipo_data, 
                avg_probability, 
                consensus
            )

            # Display results
            print("\n===== Prediction Results =====")
            print(f"IPO Name: {ipo_data['IPOName']}")
            print(f"Issue Size: ₹{ipo_data['Issue_Size']} Cr")
            print(f"Issue Price: ₹{ipo_data['Issue_Price']}")
            print(f"Subscription Details:")
            print(f"  - QIB: {ipo_data['Subscription_QIB']}x")
            print(f"  - HNI: {ipo_data['Subscription_HNI']}x")
            print(f"  - RII: {ipo_data['Subscription_RII']}x")
            print(f"  - Total: {ipo_data['Subscription_Total']}x")
            
            print("\nModel Predictions:")
            for model_result in result:
                print(f"{model_result['model']}: {model_result['status']} (Confidence: {model_result['probability']*100:.2f}%)")

            print(f"\nConsensus Prediction: {consensus}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Expected Listing {'Gain' if listing_gain_percent > 0 else 'Loss'}: {listing_gain_percent:.2f}%")
            print(f"Predicted Opening Price: ₹{predicted_opening_price:.2f}")

             # Calculate absolute profit/loss per share
            absolute_change = predicted_opening_price - ipo_data['Issue_Price']
            if absolute_change > 0:
                print(f"Expected Profit per Share: ₹{absolute_change:.2f}")
            else:
                print(f"Expected Loss per Share: ₹{abs(absolute_change):.2f}")
            
            
            # Estimate listing gain based on probability
            if consensus == "PROFITABLE":
                estimated_gain = (avg_probability - 0.5) * 2 * 30  # Simple linear scaling
                print(f"Estimated Listing Gain: {estimated_gain:.2f}%")
            else:
                estimated_loss = (0.5 - avg_probability) * 2 * 15  # Simple linear scaling
                print(f"Estimated Listing Loss: {estimated_loss:.2f}%")
            
            # Ask if user wants to continue
            choice = input("\nDo you want to predict another IPO? (y/n): ")
            if choice.lower() != 'y':
                break
                
    except (FileNotFoundError, IOError):
        print("Models not found. Training new models...")
        
        # If models aren't found, ask user for the dataset path
        filepath = input("Enter the path to your CSV dataset: ")
        
        try:
            # Load data and train models
            df, ipo_name_df, listing_gains_percent = load_and_preprocess_data(filepath)
            
            # Prepare data for modeling
            X_train_res, X_test_scaled, y_train_res, y_test, selected_features, scaler = prepare_data(df)
            
            # Train ensemble model
            model = train_ensemble_models(X_train_res, y_train_res, X_test_scaled, y_test)
            
            # Save model
            joblib.dump(model, 'ipo_model_ensemble.pkl')
            print("Model trained and saved successfully!")
            
            # Now continue with prediction
            main()
            
        except Exception as e:
            print(f"Error training models: {e}")
            return
    
    print("\n===== Session Complete =====")

if __name__ == "__main__":
    predict_ipo_interactive()
    main()