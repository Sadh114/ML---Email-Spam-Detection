import os
import sys
import importlib
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

class ProjectTester:
    def __init__(self):
        self.test_results = []
        self.errors = []
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        
        if not passed:
            self.errors.append(f"{test_name}: {message}")
    
    def test_dependencies(self):
        """Test if all required packages are installed"""
        print("\n" + "="*50)
        print("TESTING DEPENDENCIES")
        print("="*50)
        
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'nltk',
            'wordcloud', 'tensorflow', 'sklearn', 'flask'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                self.log_test(f"Import {package}", True)
            except ImportError as e:
                self.log_test(f"Import {package}", False, str(e))
    
    def test_files_exist(self):
        """Test if all required files exist"""
        print("\n" + "="*50)
        print("TESTING FILE EXISTENCE")
        print("="*50)
        
        required_files = [
            'requirements.txt',
            'train_model.py', 
            'predict.py',
            'app.py',
            'config.py',
            'batch_process.py',
            'Emails.csv'
        ]
        
        for file in required_files:
            exists = os.path.exists(file)
            self.log_test(f"File exists: {file}", exists, 
                         "" if exists else "File not found")
    
    def test_dataset(self):
        """Test dataset format and content"""
        print("\n" + "="*50)
        print("TESTING DATASET")
        print("="*50)
        
        try:
            df = pd.read_csv('Emails.csv')
            
            # Check columns
            required_columns = ['text', 'label']
            for col in required_columns:
                has_column = col in df.columns
                self.log_test(f"Dataset has '{col}' column", has_column)
            
            # Check labels
            if 'label' in df.columns:
                unique_labels = set(df['label'].unique())
                expected_labels = {'spam', 'ham'}
                valid_labels = expected_labels.issubset(unique_labels) or unique_labels.issubset(expected_labels)
                self.log_test("Dataset has valid labels (spam/ham)", valid_labels,
                             f"Found labels: {list(unique_labels)}")
            
            # Check size
            min_size = 100
            has_enough_data = len(df) >= min_size
            self.log_test(f"Dataset has at least {min_size} samples", has_enough_data,
                         f"Found {len(df)} samples")
            
            # Check for missing values in text
            if 'text' in df.columns:
                missing_text = df['text'].isna().sum()
                no_missing = missing_text == 0
                self.log_test("No missing text values", no_missing,
                             f"Found {missing_text} missing values" if not no_missing else "")
                
        except Exception as e:
            self.log_test("Load dataset", False, str(e))
    
    def test_nltk_data(self):
        """Test NLTK data availability"""
        print("\n" + "="*50)
        print("TESTING NLTK DATA")
        print("="*50)
        
        try:
            import nltk
            from nltk.corpus import stopwords
            
            # Try to access stopwords
            try:
                english_stopwords = set(stopwords.words('english'))
                has_stopwords = len(english_stopwords) > 0
                self.log_test("NLTK stopwords available", has_stopwords,
                             f"Found {len(english_stopwords)} stopwords")
            except:
                # Try to download
                try:
                    nltk.download('stopwords', quiet=True)
                    english_stopwords = set(stopwords.words('english'))
                    self.log_test("NLTK stopwords downloaded", True)
                except Exception as e:
                    self.log_test("NLTK stopwords", False, str(e))
                    
        except ImportError:
            self.log_test("NLTK import", False, "NLTK not installed")
    
    def test_model_training(self):
        """Test if model can be trained (basic syntax check)"""
        print("\n" + "="*50)
        print("TESTING MODEL TRAINING SCRIPT")
        print("="*50)
        
        try:
            # Test if script can be imported without errors
            import train_model
            self.log_test("Import train_model.py", True)
            
            # Test if main functions exist
            required_functions = ['load_data', 'preprocess_text', 'create_model']
            for func_name in required_functions:
                has_function = hasattr(train_model, func_name)
                self.log_test(f"Function exists: {func_name}", has_function)
                
        except Exception as e:
            self.log_test("Import train_model.py", False, str(e))
    
    def test_prediction_module(self):
        """Test prediction module"""
        print("\n" + "="*50)
        print("TESTING PREDICTION MODULE")
        print("="*50)
        
        try:
            import predict
            self.log_test("Import predict.py", True)
            
            # Test if SpamDetector class exists
            has_detector = hasattr(predict, 'SpamDetector')
            self.log_test("SpamDetector class exists", has_detector)
            
        except Exception as e:
            self.log_test("Import predict.py", False, str(e))
    
    def test_flask_app(self):
        """Test Flask app structure"""
        print("\n" + "="*50)
        print("TESTING FLASK APPLICATION")
        print("="*50)
        
        try:
            import app
            self.log_test("Import app.py", True)
            
            # Test if Flask app exists
            has_app = hasattr(app, 'app')
            self.log_test("Flask app object exists", has_app)
            
        except Exception as e:
            self.log_test("Import app.py", False, str(e))
    
    def test_config(self):
        """Test configuration module"""
        print("\n" + "="*50)
        print("TESTING CONFIGURATION")
        print("="*50)
        
        try:
            import config
            self.log_test("Import config.py", True)
            
            # Test if Config class exists
            has_config = hasattr(config, 'Config')
            self.log_test("Config class exists", has_config)
            
            if has_config:
                # Test some key attributes
                key_attrs = ['DATA_FILE', 'MODEL_DIR', 'MAX_SEQUENCE_LENGTH']
                for attr in key_attrs:
                    has_attr = hasattr(config.Config, attr)
                    self.log_test(f"Config has {attr}", has_attr)
                    
        except Exception as e:
            self.log_test("Import config.py", False, str(e))
    
    def test_environment(self):
        """Test Python environment"""
        print("\n" + "="*50)
        print("TESTING ENVIRONMENT")
        print("="*50)
        
        # Python version
        python_version = sys.version_info
        version_ok = python_version >= (3, 7)
        self.log_test("Python version >= 3.7", version_ok,
                     f"Found Python {python_version.major}.{python_version.minor}")
        
        # Memory check (basic)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_ok = memory_gb >= 2
            self.log_test("Available RAM >= 2GB", memory_ok,
                         f"Found {memory_gb:.1f} GB RAM")
        except ImportError:
            self.log_test("RAM check", True, "psutil not available - skipped")
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*60)
        print("SPAM DETECTION PROJECT - AUTOMATED TESTING")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        self.test_environment()
        self.test_dependencies()
        self.test_files_exist()
        self.test_dataset()
        self.test_nltk_data()
        self.test_config()
        self.test_model_training()
        self.test_prediction_module()
        self.test_flask_app()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS ({failed_tests}):")
            for error in self.errors:
                print(f"  • {error}")
            
            print(f"\n🔧 NEXT STEPS:")
            print("1. Install missing dependencies: pip install -r requirements.txt")
            print("2. Download NLTK data: python -c \"import nltk; nltk.download('stopwords')\"")
            print("3. Ensure Emails.csv is in the project directory")
            print("4. Check file permissions and Python version")
        else:
            print("\n🎉 ALL TESTS PASSED!")
            print("Your spam detection project is ready to use!")
            print("\nNext steps:")
            print("1. Run: python train_model.py")
            print("2. Run: python predict.py") 
            print("3. Run: python app.py")

def create_sample_test_data():
    """Create sample test data for testing"""
    if not os.path.exists('test_emails.txt'):
        test_emails = [
            "Win $1000000 now! Click here immediately!",
            "Hi John, can we meet for coffee tomorrow?",
            "URGENT: Your account will be suspended!",
            "Thank you for your presentation today.",
            "FREE PILLS! No prescription needed!"
        ]
        
        with open('test_emails.txt', 'w') as f:
            f.write('\n'.join(test_emails))
        
        print("Created test_emails.txt for testing")

def main():
    """Main testing function"""
    # Create sample test data
    create_sample_test_data()
    
    # Run tests
    tester = ProjectTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()