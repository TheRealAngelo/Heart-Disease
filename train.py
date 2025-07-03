"""
Training script for Heart Disease Predictor
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Load and prepare the heart disease dataset"""
    # For demo purposes, create synthetic data
    # In a real scenario, you would load from heart_disease.csv
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic features
    features = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 
                'MentalHealth', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity',
                'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good',
                'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']
    
    # Generate realistic data
    data = {}
    data['BMI'] = np.random.normal(26, 4, n_samples)
    data['Smoking'] = np.random.binomial(1, 0.15, n_samples)
    data['AlcoholDrinking'] = np.random.binomial(1, 0.05, n_samples)
    data['Stroke'] = np.random.binomial(1, 0.04, n_samples)
    data['PhysicalHealth'] = np.random.poisson(3, n_samples)
    data['MentalHealth'] = np.random.poisson(3, n_samples)
    data['DiffWalking'] = np.random.binomial(1, 0.13, n_samples)
    data['Sex'] = np.random.binomial(1, 0.5, n_samples)
    data['Diabetic'] = np.random.binomial(1, 0.08, n_samples)
    data['PhysicalActivity'] = np.random.binomial(1, 0.75, n_samples)
    data['GenHealth_Fair'] = np.random.binomial(1, 0.15, n_samples)
    data['GenHealth_Good'] = np.random.binomial(1, 0.30, n_samples)
    data['GenHealth_Poor'] = np.random.binomial(1, 0.05, n_samples)
    data['GenHealth_Very good'] = np.random.binomial(1, 0.25, n_samples)
    data['SleepTime'] = np.random.normal(7, 1, n_samples)
    data['Asthma'] = np.random.binomial(1, 0.09, n_samples)
    data['KidneyDisease'] = np.random.binomial(1, 0.03, n_samples)
    data['SkinCancer'] = np.random.binomial(1, 0.05, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate realistic target based on risk factors
    y = []
    for i in range(n_samples):
        risk_score = 0
        if df.iloc[i]['Smoking'] == 1:
            risk_score += 0.3
        if df.iloc[i]['Diabetic'] == 1:
            risk_score += 0.25
        if df.iloc[i]['Stroke'] == 1:
            risk_score += 0.35
        if df.iloc[i]['BMI'] > 30:
            risk_score += 0.2
        if df.iloc[i]['PhysicalHealth'] > 10:
            risk_score += 0.15
        if df.iloc[i]['PhysicalActivity'] == 0:
            risk_score += 0.1
        
        # Add some randomness
        risk_score += np.random.normal(0, 0.1)
        y.append(1 if risk_score > 0.4 else 0)
    
    df['HeartDisease'] = y
    
    return df, features

def train_model():
    """Train the heart disease prediction model"""
    print("Loading data...")
    df, features = load_data()
    
    # Prepare features and target
    X = df[features].values
    y = df['HeartDisease'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Heart Disease cases: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/heart_disease_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(features, 'models/features.joblib')
    
    print("\nModel saved successfully!")
    print("Files created:")
    print("- models/heart_disease_model.joblib")
    print("- models/scaler.joblib")
    print("- models/features.joblib")
    
    return model, scaler, features

if __name__ == "__main__":
    train_model()
