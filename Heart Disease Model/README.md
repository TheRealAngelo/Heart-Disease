# Heart Disease Risk Predictor

A machine learning-powered web application for predicting heart disease risk based on patient information.

## Features

- **Interactive Web Interface**: Easy-to-use form for inputting patient data
- **Real-time Prediction**: Instant heart disease risk assessment
- **Confidence Scoring**: Shows prediction confidence levels
- **Risk Factor Analysis**: Identifies key risk factors from input data
- **Visual Dashboard**: Interactive charts and gauges for results
- **Responsive Design**: Works on desktop and mobile devices

## Installation

### Local Development

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Download the heart disease dataset (e.g., from Kaggle)
   - Place it in the project directory as `heart_2020_uncleaned.csv`
   - Or update the path in `train_model.py`

4. **Train the model**:
   ```bash
   python train_model.py
   ```

5. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   - Create a GitHub repository
   - Push all files to the repository
   - Make sure the repository is public

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Pre-deployment Setup**:
   - Make sure you have the trained model files (`*.pkl`) in your repository
   - Or set up the training to run automatically on first deployment

### Option 2: Heroku

1. **Create additional files**:

   `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

   `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

2. **Deploy to Heroku**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py"]
   ```

2. **Build and run**:
   ```bash
   docker build -t heart-disease-app .
   docker run -p 8501:8501 heart-disease-app
   ```

## File Structure

```
Heart Disease Model/
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── heart_disease_model.pkl         # Trained model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── features.pkl                    # Feature names (generated)
└── heart_2020_uncleaned.csv       # Dataset (you need to provide)
```

## Usage

1. **Access the application** through your web browser
2. **Fill in patient information** in the sidebar:
   - Demographics (age, sex)
   - Physical measurements (BMI)
   - Health conditions (diabetes, stroke, etc.)
   - Lifestyle factors (smoking, exercise, sleep)
   - Medical history
3. **Click "Predict Risk"** to get the assessment
4. **Review results**:
   - Risk level (High/Low)
   - Confidence score
   - Risk factors analysis
   - Visual risk gauge

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 15+ patient characteristics
- **Preprocessing**: 
  - Missing value imputation
  - Feature scaling with StandardScaler
  - SMOTE for class balancing
  - One-hot encoding for categorical variables
- **Performance**: Optimized for balanced accuracy

## Important Disclaimers

⚠️ **Medical Disclaimer**: This application is for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Technical Requirements

- Python 3.7+
- 4GB RAM minimum for model training
- Internet connection for package installation

## Support

For issues or questions:
1. Check that all dependencies are installed correctly
2. Ensure the dataset is in the correct format
3. Verify all model files are generated properly
4. Check the Streamlit logs for error messages

## License

This project is for educational purposes. Please ensure you have appropriate permissions for any datasets used.
