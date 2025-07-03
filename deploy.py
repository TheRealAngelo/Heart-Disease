"""
Deployment script for Heart Disease Predictor
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_app():
    """Run the Streamlit app"""
    print("Starting Heart Disease Predictor...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])

def setup_project():
    """Set up the project for first time use"""
    print("Setting up Heart Disease Predictor...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Install requirements
    install_requirements()
    
    # Create demo model if needed
    from utils import create_demo_model
    if not os.path.exists('models/heart_disease_model.joblib'):
        print("Creating demo model...")
        create_demo_model()
        print("Demo model created successfully!")
    
    print("Setup complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "setup":
            setup_project()
        elif command == "run":
            run_app()
        elif command == "install":
            install_requirements()
        else:
            print("Usage: python deploy.py [setup|run|install]")
    else:
        print("Usage: python deploy.py [setup|run|install]")
        print("  setup   - Set up project for first time")
        print("  run     - Run the Streamlit app")
        print("  install - Install requirements only")
