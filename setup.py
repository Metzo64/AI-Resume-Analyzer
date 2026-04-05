"""
Setup Script for AI Resume Analyzer
Checks dependencies and prepares the environment
"""

import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("🐍 Checking Python Version")
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("📦 Checking Dependencies")
    
    required_packages = [
        'flask',
        'pandas',
        'sklearn',
        'torch',
        'transformers',
        'sentence_transformers',
        'nltk'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed")
    return True

def check_data_files():
    """Check if required data files exist"""
    print_header("📁 Checking Data Files")
    
    required_files = [
        'data/Resume.csv',
        'data/job_title_des.csv'
    ]
    
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        print("Please ensure all data files are in the data/ directory")
        return False
    
    print("\n✅ All data files are present")
    return True

def check_directories():
    """Create required directories if they don't exist"""
    print_header("📂 Checking Directories")
    
    required_dirs = [
        'models',
        'data',
        'modules',
        'static',
        'templates'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ - Creating...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {dir_path}/ - Created")
    
    print("\n✅ All directories are ready")
    return True

def check_model():
    """Check if trained model exists"""
    print_header("🧠 Checking Trained Model")
    
    model_path = 'models/resume_classifier'
    
    if os.path.exists(model_path) and os.path.isdir(model_path):
        files = os.listdir(model_path)
        if files:
            print(f"✅ Model found at {model_path}/")
            print(f"   Files: {len(files)} model files")
            return True
    
    print(f"⚠️  No trained model found at {model_path}/")
    print("\nTo train the model, run:")
    print("   python train_model.py")
    print("\nNote: Training takes 30-60 minutes on CPU")
    return False

def download_nltk_data():
    """Download required NLTK data"""
    print_header("📚 Downloading NLTK Data")
    
    try:
        import nltk
        
        print("Downloading 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)
        print("✅ punkt")
        
        print("Downloading 'stopwords'...")
        nltk.download('stopwords', quiet=True)
        print("✅ stopwords")
        
        print("\n✅ NLTK data downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {str(e)}")
        return False

def run_tests():
    """Run module tests"""
    print_header("🧪 Running Module Tests")
    
    print("Running test_modules.py...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'test_modules.py'],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("🚀 AI RESUME ANALYZER - SETUP")
    print("="*70)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Data Files': check_data_files(),
        'Directories': check_directories(),
        'NLTK Data': download_nltk_data(),
        'Trained Model': check_model()
    }
    
    print_header("📊 SETUP SUMMARY")
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {check_name}")
    
    print("\n" + "="*70)
    
    # Determine next steps
    all_critical_passed = all([
        checks['Python Version'],
        checks['Dependencies'],
        checks['Data Files'],
        checks['Directories']
    ])
    
    if all_critical_passed:
        print("\n✅ Setup completed successfully!")
        
        if not checks['Trained Model']:
            print("\n📝 NEXT STEPS:")
            print("   1. Train the model: python train_model.py")
            print("   2. Run the application: python app.py")
        else:
            print("\n📝 NEXT STEP:")
            print("   Run the application: python app.py")
            print("\n   Or run tests: python test_modules.py")
        
        print("\n🌐 The app will be available at: http://localhost:5000")
    else:
        print("\n⚠️  Setup incomplete. Please fix the issues above.")
        print("\n📝 COMMON FIXES:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Ensure data files are in data/ directory")
    
    print("\n" + "="*70 + "\n")
    
    return all_critical_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
