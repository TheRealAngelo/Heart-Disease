"""
Train and save the heart disease prediction model
This script extracts the training code from the notebook and saves the required model files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter
import os

def train_model():
    """Train the heart disease prediction model"""
    
    # Check if dataset exists - you'll need to provide the correct path
    dataset_path = 'heart_2020_uncleaned.csv'  # Update this path as needed
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please download the heart disease dataset and place it in the same directory")
        print("Or update the dataset_path variable in this script")
        return False
    
    print("Loading dataset...")
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with shape: {df.shape}")
    
    print("Preprocessing data...")
    # Drop columns with too many missing values
    df = df.loc[:, df.isnull().mean() < 0.4]
    
    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    
    # Impute numeric and categorical
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # Set your target column
    target_column = 'HeartDisease'  # Change if needed
    
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Applying SMOTE for class balancing...")
    print("Before SMOTE:", Counter(y))
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print("After SMOTE:", Counter(y_resampled))
    
    # Train-test split AFTER resampling
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    # Use class_weight='balanced'
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model files...")
    # Save model, scaler, and column names
    joblib.dump(model, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'features.pkl')
    
    print("Model training completed successfully!")
    print("Saved files:")
    print("- heart_disease_model.pkl")
    print("- scaler.pkl") 
    print("- features.pkl")
    
    return True

if __name__ == "__main__":
    success = train_model()
    if success:
        print("\n Model training completed! You can now run the Streamlit app.")
    else:
        print("\n Model training failed. Please check the error messages above.")
