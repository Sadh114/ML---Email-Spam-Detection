import os
import json

class Config:
    """Configuration settings for the spam detection project"""
    
    # Data settings
    DATA_FILE = 'Emails.csv'
    MODEL_DIR = 'models'
    
    # Model parameters
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 32
    LSTM_UNITS = 16
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.5
    RECURRENT_DROPOUT_RATE = 0.2
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EPOCHS = 20
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.0  # Use test set for validation
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MONITOR = 'val_accuracy'
    
    # Learning rate reduction
    LR_REDUCTION_PATIENCE = 2
    LR_REDUCTION_FACTOR = 0.5
    LR_REDUCTION_MONITOR = 'val_loss'
    
    # Prediction settings
    PREDICTION_THRESHOLD = 0.5
    
    # WordCloud settings
    WORDCLOUD_MAX_WORDS = 100
    WORDCLOUD_WIDTH = 800
    WORDCLOUD_HEIGHT = 400
    WORDCLOUD_BACKGROUND_COLOR = 'black'
    WORDCLOUD_COLORMAP = 'viridis'
    
    # Flask settings
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    FLASK_DEBUG = True
    MAX_BATCH_SIZE = 100
    
    # File paths
    MODEL_FILE = 'spam_detection_model.h5'
    TOKENIZER_FILE = 'tokenizer.pkl'
    
    @classmethod
    def get_model_path(cls):
        return os.path.join(cls.MODEL_DIR, cls.MODEL_FILE)
    
    @classmethod
    def get_tokenizer_path(cls):
        return os.path.join(cls.MODEL_DIR, cls.TOKENIZER_FILE)
    
    @classmethod
    def ensure_model_dir(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            'data_file': cls.DATA_FILE,
            'model_dir': cls.MODEL_DIR,
            'max_sequence_length': cls.MAX_SEQUENCE_LENGTH,
            'embedding_dim': cls.EMBEDDING_DIM,
            'lstm_units': cls.LSTM_UNITS,
            'dense_units': cls.DENSE_UNITS,
            'dropout_rate': cls.DROPOUT_RATE,
            'test_size': cls.TEST_SIZE,
            'random_state': cls.RANDOM_STATE,
            'epochs': cls.EPOCHS,
            'batch_size': cls.BATCH_SIZE,
            'prediction_threshold': cls.PREDICTION_THRESHOLD,
            'flask_host': cls.FLASK_HOST,
            'flask_port': cls.FLASK_PORT
        }
    
    @classmethod
    def save_config(cls, filepath='config.json'):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(cls.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath='config.json'):
        """Load configuration from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Update class attributes
            for key, value in config_dict.items():
                if hasattr(cls, key.upper()):
                    setattr(cls, key.upper(), value)
            
            print(f"Configuration loaded from {filepath}")
        else:
            print(f"Config file {filepath} not found. Using default settings.")

if __name__ == "__main__":
    # Create and save default configuration
    Config.save_config()
    print("\nDefault configuration:")
    for key, value in Config.to_dict().items():
        print(f"{key}: {value}")