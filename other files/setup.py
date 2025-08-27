#!/usr/bin/env python3
"""
Email Spam Detection Project - Automated Setup Script
This script automates the entire setup process for the spam detection project.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import json
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, description):
    """Print step information"""
    print(f"\n🔄 Step {step_num}: {description}")
    print("-" * 40)

def run_command(command, description="", check=True):
    """Run a command and handle errors"""
    try:
        if description:
            print(f"Running: {description}")
        
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        return True, result.stdout, result.stderr
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected.")
        print("❌ This project requires Python 3.7 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} detected - Compatible!")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    venv_name = "spam_env"
    
    if os.path.exists(venv_name):
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return True
    
    print(f"Creating virtual environment: {venv_name}")
    success, stdout, stderr = run_command(f"python -m venv {venv_name}")
    
    if success:
        print(f"✅ Virtual environment '{venv_name}' created successfully!")
        print("\n📝 To activate it manually:")
        if sys.platform == "win32":
            print(f"   {venv_name}\\Scripts\\activate")
        else:
            print(f"   source {venv_name}/bin/activate")
    
    return success

def install_dependencies():
    """Install required Python packages"""
    print("Installing required packages...")
    
    # Define activation command based on OS
    if sys.platform == "win32":
        activate_cmd = "spam_env\\Scripts\\activate"
        pip_cmd = "spam_env\\Scripts\\pip"
    else:
        activate_cmd = "source spam_env/bin/activate"
        pip_cmd = "spam_env/bin/pip"
    
    # Try to install using pip from virtual environment
    packages = [
        "numpy", "pandas", "matplotlib", "seaborn", "nltk",
        "wordcloud", "tensorflow", "scikit-learn", "flask"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        success, stdout, stderr = run_command(f"{pip_cmd} install {package}")
        if not success:
            print(f"❌ Failed to install {package}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK stopwords...")
    
    if sys.platform == "win32":
        python_cmd = "spam_env\\Scripts\\python"
    else:
        python_cmd = "spam_env/bin/python"
    
    cmd = f'{python_cmd} -c "import nltk; nltk.download(\'stopwords\', quiet=True)"'
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ NLTK stopwords downloaded!")
    else:
        print("❌ Failed to download NLTK data")
    
    return success

def verify_files():
    """Verify all project files exist"""
    required_files = [
        "requirements.txt", "README.md", "train_model.py", 
        "predict.py", "app.py", "config.py", "batch_process.py",
        "test_project.py", "Emails.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files are present!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["models", "logs", "output"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory created/verified: {directory}")

def run_tests():
    """Run project tests"""
    print("Running project tests...")
    
    if sys.platform == "win32":
        python_cmd = "spam_env\\Scripts\\python"
    else:
        python_cmd = "spam_env/bin/python"
    
    success, stdout, stderr = run_command(f"{python_cmd} test_project.py")
    return success

def create_run_scripts():
    """Create convenient run scripts"""
    
    if sys.platform == "win32":
        # Windows batch files
        scripts = {
            "run_training.bat": """@echo off
echo Starting Spam Detection Model Training...
spam_env\\Scripts\\python train_model.py
pause""",
            
            "run_prediction.bat": """@echo off
echo Starting Spam Detection Prediction Interface...
spam_env\\Scripts\\python predict.py
pause""",
            
            "run_webapp.bat": """@echo off
echo Starting Spam Detection Web Application...
echo Open your browser and go to: http://localhost:5000
spam_env\\Scripts\\python app.py
pause""",
            
            "run_tests.bat": """@echo off
echo Running Project Tests...
spam_env\\Scripts\\python test_project.py
pause"""
        }
    else:
        # Unix shell scripts
        scripts = {
            "run_training.sh": """#!/bin/bash
echo "Starting Spam Detection Model Training..."
source spam_env/bin/activate
python train_model.py""",
            
            "run_prediction.sh": """#!/bin/bash
echo "Starting Spam Detection Prediction Interface..."
source spam_env/bin/activate
python predict.py""",
            
            "run_webapp.sh": """#!/bin/bash
echo "Starting Spam Detection Web Application..."
echo "Open your browser and go to: http://localhost:5000"
source spam_env/bin/activate
python app.py""",
            
            "run_tests.sh": """#!/bin/bash
echo "Running Project Tests..."
source spam_env/bin/activate
python test_project.py"""
        }
    
    for filename, content in scripts.items():
        with open(filename, 'w') as f:
            f.write(content)
        
        # Make executable on Unix systems
        if not sys.platform == "win32":
            os.chmod(filename, 0o755)
        
        print(f"✅ Created: {filename}")

def main():
    """Main setup function"""
    print_header("SPAM DETECTION PROJECT - AUTOMATED SETUP")
    print("This script will set up everything you need to run the spam detection project.")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python Version")
    if not check_python_version():
        return
    
    # Step 2: Verify files
    print_step(2, "Verifying Project Files")
    if not verify_files():
        print("❌ Some required files are missing. Please ensure all project files are present.")
        return
    
    # Step 3: Create virtual environment
    print_step(3, "Creating Virtual Environment")
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        return
    
    # Step 4: Install dependencies
    print_step(4, "Installing Dependencies")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Step 5: Download NLTK data
    print_step(5, "Downloading NLTK Data")
    download_nltk_data()
    
    # Step 6: Create directories
    print_step(6, "Creating Directories")
    create_directories()
    
    # Step 7: Create run scripts
    print_step(7, "Creating Convenience Scripts")
    create_run_scripts()
    
    # Step 8: Run tests
    print_step(8, "Running Project Tests")
    tests_passed = run_tests()
    
    # Final summary
    print_header("SETUP COMPLETE!")
    
    if tests_passed:
        print("🎉 SUCCESS! Your spam detection project is ready to use!")
    else:
        print("⚠️  Setup completed but some tests failed. Check the test output above.")
    
    print("\n📋 WHAT'S NEXT:")
    print("1. Train the model:")
    if sys.platform == "win32":
        print("   • Double-click: run_training.bat")
        print("   • Or run: spam_env\\Scripts\\python train_model.py")
    else:
        print("   • Run: ./run_training.sh")
        print("   • Or run: source spam_env/bin/activate && python train_model.py")
    
    print("\n2. Test predictions:")
    if sys.platform == "win32":
        print("   • Double-click: run_prediction.bat")
    else:
        print("   • Run: ./run_prediction.sh")
    
    print("\n3. Start web application:")
    if sys.platform == "win32":
        print("   • Double-click: run_webapp.bat")
    else:
        print("   • Run: ./run_webapp.sh")
    print("   • Open browser: http://localhost:5000")
    
    print("\n📚 DOCUMENTATION:")
    print("   • README.md - Project overview and usage")
    print("   • setup-instructions.md - Detailed setup guide")
    
    print("\n🧪 TESTING:")
    if sys.platform == "win32":
        print("   • Run tests: run_tests.bat")
    else:
        print("   • Run tests: ./run_tests.sh")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        print("Please check the error message above and try running the setup again.")