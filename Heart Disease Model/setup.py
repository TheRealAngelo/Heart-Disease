#!/usr/bin/env python3
"""
Quick setup script for the Heart Disease Predictor app
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f" {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f" {description} found")
        return True
    else:
        print(f" {description} not found at {filepath}")
        return False

def main():
    print(" Heart Disease Predictor Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print(" app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print(" Python 3.7+ is required")
        sys.exit(1)
    print(f" Python {python_version.major}.{python_version.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("  Dependency installation failed. You may need to install packages manually.")
    
    # Check for dataset
    dataset_found = False
    possible_paths = [
        "heart_2020_uncleaned.csv",
        "heart_disease_data.csv",
        "heart.csv"
    ]
    
    for path in possible_paths:
        if check_file_exists(path, f"Dataset ({path})"):
            dataset_found = True
            break
    
    if not dataset_found:
        print("\n  Dataset not found. Please:")
        print("1. Download a heart disease dataset (e.g., from Kaggle)")
        print("2. Place it in this directory as 'heart_2020_uncleaned.csv'")
        print("3. Or update the path in 'train_model.py'")
        print("\nYou can continue without the dataset, but you'll need model files.")
    
    # Check for model files
    model_files = [
        "heart_disease_model.pkl",
        "scaler.pkl", 
        "features.pkl"
    ]
    
    model_files_exist = all(os.path.exists(f) for f in model_files)
    
    if model_files_exist:
        print(" All model files found")
    elif dataset_found:
        print("\n Training model...")
        if run_command("python train_model.py", "Model training"):
            print(" Model training completed")
        else:
            print(" Model training failed")
            return False
    else:
        print(" Model files not found and no dataset available for training")
        return False
    
    print("\n Setup completed successfully!")
    print("\nTo run the app:")
    print("  streamlit run app.py")
    print("\nTo deploy:")
    print("  1. Push to GitHub")
    print("  2. Deploy on Streamlit Cloud (share.streamlit.io)")
    print("  3. Or follow deployment instructions in README.md")
    
    # Ask if user wants to run the app now
    response = input("\nWould you like to run the app now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("\n Starting Streamlit app...")
        try:
            subprocess.run("streamlit run app.py", shell=True)
        except KeyboardInterrupt:
            print("\n App stopped by user")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
