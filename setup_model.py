"""
Setup script to create model files for Streamlit Cloud deployment
This runs automatically when the app starts if model files don't exist
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def create_model_if_not_exists():
    """Create model files if they don't exist (for Streamlit Cloud deployment)"""
    
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'heart_disease_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    features_path = os.path.join(model_dir, 'features.pkl')
    
    # Check if model files exist
    if all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
        return True
    
    print("Creating model files for deployment...")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic but realistic data
    n_samples = 5000
    
    # Feature names
    feature_names = [
        'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
        'MentalHealth', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity',
        'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer',
        'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good'
    ]
    
    # Create realistic synthetic data
    data = {}
    data['BMI'] = np.clip(np.random.normal(27, 5, n_samples), 15, 45)
    data['Smoking'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['AlcoholDrinking'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    data['Stroke'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    data['PhysicalHealth'] = np.clip(np.random.poisson(5, n_samples), 0, 30)
    data['MentalHealth'] = np.clip(np.random.poisson(4, n_samples), 0, 30)
    data['DiffWalking'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['Sex'] = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    data['Diabetic'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['PhysicalActivity'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    data['SleepTime'] = np.clip(np.random.normal(7.5, 1.5, n_samples), 3, 12)
    data['Asthma'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    data['KidneyDisease'] = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    data['SkinCancer'] = np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
    
    # General Health dummy variables
    gen_health_categories = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.15, 0.25, 0.15, 0.20, 0.25])
    data['GenHealth_Fair'] = (gen_health_categories == 0).astype(int)
    data['GenHealth_Good'] = (gen_health_categories == 1).astype(int)
    data['GenHealth_Poor'] = (gen_health_categories == 2).astype(int)
    data['GenHealth_Very good'] = (gen_health_categories == 3).astype(int)
    
    # Create feature matrix
    X = np.column_stack([data[feature] for feature in feature_names])
    
    # Create realistic target variable
    risk_score = (
        1.5 * data['Smoking'] +
        0.15 * np.maximum(0, data['BMI'] - 25) +
        1.2 * data['Diabetic'] +
        2.0 * data['Stroke'] +
        0.08 * data['PhysicalHealth'] +
        0.05 * data['MentalHealth'] +
        1.0 * data['DiffWalking'] +
        -0.6 * data['PhysicalActivity'] +
        1.5 * data['GenHealth_Poor'] +
        0.8 * data['GenHealth_Fair'] +
        0.8 * data['KidneyDisease'] +
        0.3 * data['Asthma'] +
        -0.15 * np.maximum(0, 8 - data['SleepTime']) +
        0.3 * data['AlcoholDrinking'] +
        0.2 * data['Sex'] +
        np.random.normal(0, 0.4, n_samples)
    )
    
    # Convert to binary (top 15% are high risk)
    threshold = np.percentile(risk_score, 85)
    y = (risk_score > threshold).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save all components
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)
    
    print("✅ Model files created successfully for deployment!")
    return True

if __name__ == "__main__":
    create_model_if_not_exists()
