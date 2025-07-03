"""
Utility functions for the Heart Disease Predictor
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_model_components():
    """Load the trained model, scaler, and feature names"""
    try:
        # Load model files
        model = joblib.load('models/heart_disease_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        features = joblib.load('models/features.joblib')
        return model, scaler, features
    except FileNotFoundError:
        print("Model files not found. Creating demo model...")
        return create_demo_model()

def create_demo_model():
    """Create a basic model for deployment"""
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Define features
    features = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 
                'MentalHealth', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity',
                'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good',
                'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']
    
    # Create realistic demo data
    np.random.seed(42)
    X_dummy = np.random.rand(1000, len(features))
    
    # Create realistic labels based on risk factors
    y_dummy = []
    for i in range(1000):
        risk_score = 0
        if X_dummy[i][1] > 0.7:  # Smoking
            risk_score += 0.3
        if X_dummy[i][8] > 0.8:  # Diabetes  
            risk_score += 0.25
        if X_dummy[i][3] > 0.8:  # Stroke
            risk_score += 0.35
        if X_dummy[i][0] > 0.8:  # High BMI
            risk_score += 0.2
        if X_dummy[i][4] > 0.7:  # Poor physical health
            risk_score += 0.15
        
        risk_score += np.random.normal(0, 0.1)
        y_dummy.append(1 if risk_score > 0.5 else 0)
    
    y_dummy = np.array(y_dummy)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    # Save model components
    joblib.dump(model, 'models/heart_disease_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(features, 'models/features.joblib')
    
    return model, scaler, features

def preprocess_input(input_data, features):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([input_data])
    
    # Handle categorical variables
    if 'GenHealth' in df.columns:
        gen_health_value = df['GenHealth'].iloc[0]
        df['GenHealth_Fair'] = 1 if gen_health_value == 'Fair' else 0
        df['GenHealth_Good'] = 1 if gen_health_value == 'Good' else 0
        df['GenHealth_Poor'] = 1 if gen_health_value == 'Poor' else 0
        df['GenHealth_Very good'] = 1 if gen_health_value == 'Very good' else 0
        df = df.drop('GenHealth', axis=1)
    
    # Ensure all features are present
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
        else:
            try:
                df[feature] = pd.to_numeric(df[feature])
            except:
                pass
    
    # Reorder columns to match training data
    df = df.reindex(columns=features, fill_value=0)
    
    return df.values
