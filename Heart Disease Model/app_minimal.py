import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.title("Heart Disease Risk Predictor")

# Simple form
age = st.slider("Age", 18, 100, 50)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoking = st.selectbox("Smoking", ["No", "Yes"])

if st.button("Predict Risk"):
    # Create a simple model on the fly
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Make prediction
    input_data = np.array([[age/100, bmi/50, 1 if smoking == "Yes" else 0]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.error(f"HIGH RISK - Probability: {probability[1]:.1%}")
    else:
        st.success(f"LOW RISK - Probability: {probability[0]:.1%}")

st.info("This is a demo app for testing deployment.")
