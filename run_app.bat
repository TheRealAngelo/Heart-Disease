@echo off
echo Starting Heart Disease Predictor...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Check if model files exist
if not exist "heart_disease_model.pkl" (
    echo Model files not found. Creating demo model...
    python create_demo_model.py
)

REM Start the app
echo Starting Streamlit app...
echo The app will open in your default browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.
streamlit run app.py

pause
