import tensorflow as tf
import pickle
import os
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

class SpamDetector:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.max_len = 100
        self.punctuations_list = string.punctuation
        self.stop_words = set(stopwords.words('english'))
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        try:
            model_path = os.path.join(self.model_dir, 'spam_detection_model.h5')
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first by running 'python train_model.py'")
            
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please train the model first.")
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("Tokenizer loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            print("Please make sure you have trained the model first by running 'python train_model.py'")
            raise
    
    def remove_punctuations(self, text):
        """Remove punctuations from text"""
        if not text:
            return ""
        temp = str.maketrans('', '', self.punctuations_list)
        return text.translate(temp)
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        if not text:
            return ""
        imp_words = [word.lower() for word in str(text).split() 
                    if word.lower() not in self.stop_words and word.strip() != '']
        return " ".join(imp_words)
    
    def preprocess_text(self, text):
        """Preprocess text the same way as during training"""
        if not text:
            return ""
        
        # Remove 'Subject' prefix
        text = text.replace('Subject', '')
        text = text.replace('subject', '')
        
        # Remove punctuations
        text = self.remove_punctuations(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        return text.strip()
    
    def predict_spam(self, email_text):
        """Predict if an email is spam or not"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(email_text)
            
            if not processed_text:
                return "not spam", 0.0, "Empty text after preprocessing"
            
            # Convert to sequence
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            
            if not sequence[0]:  # Empty sequence
                return "not spam", 0.0, "No known words found in text"
            
            # Pad sequence
            padded_sequence = pad_sequences(sequence, maxlen=self.max_len, 
                                          padding='post', truncating='post')
            
            # Make prediction
            prediction_prob = self.model.predict(padded_sequence, verbose=0)[0][0]
            
            # Convert to label
            label = 'spam' if prediction_prob >= 0.5 else 'not spam'
            
            return label, float(prediction_prob), "Success"
            
        except Exception as e:
            return "error", 0.0, f"Error during prediction: {str(e)}"
    
    def predict_batch(self, email_list):
        """Predict spam for a list of emails"""
        results = []
        for i, email in enumerate(email_list):
            label, prob, status = self.predict_spam(email)
            results.append({
                'email_index': i,
                'email_preview': email[:100] + "..." if len(email) > 100 else email,
                'prediction': label,
                'probability': prob,
                'status': status
            })
        return results

def interactive_prediction():
    """Interactive spam prediction interface"""
    print("=" * 60)
    print("EMAIL SPAM DETECTION - PREDICTION INTERFACE")
    print("=" * 60)
    print("Enter email text to classify (type 'quit' to exit)")
    print("=" * 60)
    
    try:
        detector = SpamDetector()
    except Exception as e:
        print(f"Failed to initialize spam detector: {e}")
        return
    
    while True:
        print("\n" + "-" * 40)
        email_text = input("Enter email text: ").strip()
        
        if email_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not email_text:
            print("Please enter some text.")
            continue
        
        # Make prediction
        label, probability, status = detector.predict_spam(email_text)
        
        print("\n" + "=" * 40)
        print("PREDICTION RESULTS")
        print("=" * 40)
        
        if status == "Success":
            print(f"Classification: {label.upper()}")
            print(f"Confidence: {probability:.4f}")
            
            if label == 'spam':
                confidence_text = "HIGH" if probability > 0.8 else ("MEDIUM" if probability > 0.6 else "LOW")
                print(f"Spam Confidence: {confidence_text}")
            else:
                confidence_text = "HIGH" if probability < 0.2 else ("MEDIUM" if probability < 0.4 else "LOW")
                print(f"Ham Confidence: {confidence_text}")
        else:
            print(f"Status: {status}")
        
        print("=" * 40)

def test_sample_emails():
    """Test with sample emails"""
    print("Testing with sample emails...")
    
    try:
        detector = SpamDetector()
    except Exception as e:
        print(f"Failed to initialize spam detector: {e}")
        return
    
    sample_emails = [
        "Congratulations! You've won $1000000! Click here immediately to claim your prize!",
        "Hi John, can we schedule a meeting for tomorrow at 2 PM?",
        "URGENT: Your account will be suspended unless you verify your details immediately!",
        "Thanks for your presentation today. The quarterly report looks great.",
        "FREE VIAGRA! No prescription needed. Order now with 50% discount!",
        "Please find attached the invoice for this month's services.",
        "WINNER! You are selected for our cash prize. Send your bank details to claim!",
        "The conference call is scheduled for Friday at 10 AM EST."
    ]
    
    results = detector.predict_batch(sample_emails)
    
    print("\n" + "=" * 80)
    print("SAMPLE EMAIL PREDICTIONS")
    print("=" * 80)
    
    for result in results:
        print(f"\nEmail {result['email_index'] + 1}:")
        print(f"Text: {result['email_preview']}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Probability: {result['probability']:.4f}")
        print("-" * 40)

def main():
    """Main function with menu options"""
    print("Welcome to Email Spam Detection!")
    print("\nOptions:")
    print("1. Interactive prediction")
    print("2. Test with sample emails")
    print("3. Both")
    
    try:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            interactive_prediction()
        elif choice == '2':
            test_sample_emails()
        elif choice == '3':
            test_sample_emails()
            interactive_prediction()
        else:
            print("Invalid choice. Running interactive prediction...")
            interactive_prediction()
            
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure NLTK data is available
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass
    
    main()
