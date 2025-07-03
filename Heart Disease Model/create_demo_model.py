"""
Create demo model files for testing the Streamlit app
This creates mock model files when you don't have the actual dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def create_demo_model():
    """Create a demo model with sample data"""
    print("Creating enhanced demo model files...")
    
    # Create sample feature names (based on common heart disease datasets)
    feature_names = [
        'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
        'MentalHealth', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity',
        'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer',
        'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good'
    ]
    
    # Create more realistic training data
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic data with realistic distributions
    data = {}
    
    # BMI - normal distribution around 26
    data['BMI'] = np.random.normal(26, 4, n_samples)
    data['BMI'] = np.clip(data['BMI'], 15, 50)  # Realistic BMI range
    
    # Binary features with realistic prevalence
    data['Smoking'] = np.random.binomial(1, 0.15, n_samples)  # 15% smokers
    data['AlcoholDrinking'] = np.random.binomial(1, 0.25, n_samples)  # 25% drink alcohol
    data['Stroke'] = np.random.binomial(1, 0.04, n_samples)  # 4% had stroke
    data['DiffWalking'] = np.random.binomial(1, 0.12, n_samples)  # 12% difficulty walking
    data['Sex'] = np.random.binomial(1, 0.48, n_samples)  # 48% male
    data['Diabetic'] = np.random.binomial(1, 0.11, n_samples)  # 11% diabetic
    data['PhysicalActivity'] = np.random.binomial(1, 0.73, n_samples)  # 73% physically active
    data['Asthma'] = np.random.binomial(1, 0.09, n_samples)  # 9% have asthma
    data['KidneyDisease'] = np.random.binomial(1, 0.03, n_samples)  # 3% kidney disease
    data['SkinCancer'] = np.random.binomial(1, 0.05, n_samples)  # 5% skin cancer
    
    # Health days (0-30)
    data['PhysicalHealth'] = np.random.poisson(3, n_samples)
    data['PhysicalHealth'] = np.clip(data['PhysicalHealth'], 0, 30)
    
    data['MentalHealth'] = np.random.poisson(2.5, n_samples) 
    data['MentalHealth'] = np.clip(data['MentalHealth'], 0, 30)
    
    # Sleep time
    data['SleepTime'] = np.random.normal(7.5, 1.2, n_samples)
    data['SleepTime'] = np.clip(data['SleepTime'], 4, 12)
    
    # General Health categories (one-hot encoded, Excellent is reference)
    gen_health_probs = [0.15, 0.25, 0.15, 0.20, 0.25]  # Fair, Good, Poor, Very good, Excellent
    gen_health_categories = np.random.choice(5, n_samples, p=gen_health_probs)
    
    data['GenHealth_Fair'] = (gen_health_categories == 0).astype(int)
    data['GenHealth_Good'] = (gen_health_categories == 1).astype(int)
    data['GenHealth_Poor'] = (gen_health_categories == 2).astype(int)
    data['GenHealth_Very good'] = (gen_health_categories == 3).astype(int)
    # Excellent is the reference category (all others = 0)
    
    # Convert to array
    X = np.column_stack([data[feature] for feature in feature_names])
    
    # Create target variable with realistic relationships
    # Higher risk factors: smoking, high BMI, diabetes, stroke, poor health, etc.
    risk_score = (
        1.5 * data['Smoking'] +                                   # Smoking major risk
        0.15 * np.maximum(0, data['BMI'] - 25) +                 # High BMI
        1.2 * data['Diabetic'] +                                 # Diabetes
        2.0 * data['Stroke'] +                                   # Previous stroke (major risk)
        0.08 * data['PhysicalHealth'] +                          # Poor physical health
        0.05 * data['MentalHealth'] +                            # Poor mental health
        1.0 * data['DiffWalking'] +                              # Difficulty walking
        -0.6 * data['PhysicalActivity'] +                        # Physical activity protective
        1.5 * data['GenHealth_Poor'] +                           # Poor general health
        0.8 * data['GenHealth_Fair'] +                           # Fair general health
        0.8 * data['KidneyDisease'] +                            # Kidney disease
        0.3 * data['Asthma'] +                                   # Asthma
        -0.15 * np.maximum(0, 8 - data['SleepTime']) +           # Poor sleep
        0.3 * data['AlcoholDrinking'] +                          # Alcohol drinking
        0.2 * data['Sex'] +                                      # Male slightly higher risk
        np.random.normal(0, 0.4, n_samples)                     # Random noise
    )
    
    # Convert to binary with approximately 15% positive rate (more realistic)
    threshold = np.percentile(risk_score, 85)  # Top 15% get classified as high risk
    y = (risk_score > threshold).astype(int)
    
    print(f"Generated {n_samples} samples with {y.sum()} positive cases ({y.mean():.1%} positive rate)")
    
    # Create and train model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Create scaler with the training data (as numpy array to avoid feature name warnings)
    scaler = StandardScaler()
    scaler.fit(X_train)  # X_train is already a numpy array
    
    # Save all components in the model directory
    import os
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(model, 'model/heart_disease_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(feature_names, 'model/features.pkl')
    
    print("✅ Enhanced demo model files created successfully!")
    print("Files saved in model/ directory:")
    print("- heart_disease_model.pkl")
    print("- scaler.pkl")
    print("- features.pkl")
    
    return True

if __name__ == "__main__":
    create_demo_model()
    print("\n Demo setup complete! You can now run the Streamlit app:")
    print("streamlit run app.py")
