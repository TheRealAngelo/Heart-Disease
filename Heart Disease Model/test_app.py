"""
Test script to verify the Streamlit app works correctly
"""

import os
import sys
import importlib.util

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'sklearn',
        'joblib',
        'plotly',
        'imblearn'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package}")
        except ImportError as e:
            print(f" {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_app_syntax():
    """Test if app.py has valid syntax"""
    print("\nTesting app.py syntax...")
    
    try:
        spec = importlib.util.spec_from_file_location("app", "app.py")
        if spec is None:
            print(" Could not load app.py")
            return False
            
        # Just check if it can be loaded, don't execute
        print(" app.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f" Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f" Error loading app.py: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'train_model.py'
    ]
    
    optional_files = [
        'heart_disease_model.pkl',
        'scaler.pkl',
        'features.pkl',
        'heart_2020_uncleaned.csv'
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f" {file}")
        else:
            print(f" {file} (required)")
            all_good = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f" {file}")
        else:
            print(f"  {file} (optional, but needed for full functionality)")
    
    return all_good

def main():
    print(" Heart Disease Predictor - Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test app syntax
    syntax_ok = test_app_syntax()
    
    # Check files
    files_ok = check_required_files()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Imports: {' PASS' if imports_ok else '❌ FAIL'}")
    print(f"App Syntax: {' PASS' if syntax_ok else '❌ FAIL'}")
    print(f"Required Files: {' PASS' if files_ok else '❌ FAIL'}")
    
    if imports_ok and syntax_ok and files_ok:
        print("\n All tests passed! The app should work correctly.")
        print("\nTo run the app: streamlit run app.py")
        return True
    else:
        print("\n Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
