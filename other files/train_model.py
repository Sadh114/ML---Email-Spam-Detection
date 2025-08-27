import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import os
import pickle
from nltk.corpus import stopwords
from wordcloud import WordCloud

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def setup_nltk():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords')
        print("NLTK stopwords downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def load_data(file_path='Emails.csv'):
    """Load and display basic info about the dataset"""
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head())
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please place the dataset in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def visualize_distribution(data, title="Email Label Distribution"):
    """Visualize the distribution of spam vs ham emails"""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='label')
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Label')
    
    # Add count annotations
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    label_counts = data['label'].value_counts()
    print(f"\nLabel Distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(data)*100:.1f}%)")

def balance_dataset(data):
    """Balance the dataset by downsampling the majority class"""
    print("\nBalancing dataset...")
    
    ham_msg = data[data['label'] == 'ham']
    spam_msg = data[data['label'] == 'spam']
    
    print(f"Original - Ham: {len(ham_msg)}, Spam: {len(spam_msg)}")
    
    # Downsample ham emails to match spam count
    min_count = min(len(ham_msg), len(spam_msg))
    
    if len(ham_msg) > len(spam_msg):
        ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)
        balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)
    else:
        spam_msg_balanced = spam_msg.sample(n=len(ham_msg), random_state=42)
        balanced_data = pd.concat([ham_msg, spam_msg_balanced]).reset_index(drop=True)
    
    print(f"Balanced - Total: {len(balanced_data)}")
    
    return balanced_data

def preprocess_text(data):
    """Clean and preprocess email text"""
    print("\nPreprocessing text...")
    
    # Make a copy to avoid modifying original data
    data = data.copy()
    
    # Remove 'Subject' from text
    data['text'] = data['text'].str.replace('Subject', '', regex=False)
    
    # Remove punctuations
    punctuations_list = string.punctuation
    def remove_punctuations(text):
        if pd.isna(text):
            return ""
        temp = str.maketrans('', '', punctuations_list)
        return str(text).translate(temp)
    
    data['text'] = data['text'].apply(remove_punctuations)
    
    # Remove stopwords
    def remove_stopwords(text):
        if pd.isna(text) or text == "":
            return ""
        stop_words = set(stopwords.words('english'))
        imp_words = [word.lower() for word in str(text).split() 
                    if word.lower() not in stop_words and word.strip() != '']
        return " ".join(imp_words)
    
    data['text'] = data['text'].apply(remove_stopwords)
    
    # Remove empty texts
    data = data[data['text'].str.strip() != ''].reset_index(drop=True)
    
    print(f"After preprocessing: {len(data)} emails remaining")
    return data

def plot_word_cloud(data, typ):
    """Generate and display word cloud for email type"""
    try:
        cleaned_texts = data['text'].dropna().astype(str)
        email_corpus = " ".join([text.strip() for text in cleaned_texts if text.strip() != ''])
        
        if not email_corpus:
            print(f"No text available to generate WordCloud for {typ} emails.")
            return
        
        wc = WordCloud(background_color='black', 
                      max_words=100, 
                      width=800, 
                      height=400,
                      colormap='viridis').generate(email_corpus)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f'WordCloud for {typ} Emails', fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"WordCloud generated for {typ} emails")
        
    except Exception as e:
        print(f"Error generating word cloud for {typ}: {e}")

def prepare_sequences(data, max_len=100):
    """Tokenize and pad sequences for model training"""
    print(f"\nPreparing sequences with max length: {max_len}")
    
    # Split data
    train_X, test_X, train_Y, test_Y = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
    )
    
    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}")
    
    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_X)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Convert to sequences
    train_sequences = tokenizer.texts_to_sequences(train_X)
    test_sequences = tokenizer.texts_to_sequences(test_X)
    
    # Pad sequences
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Convert labels to binary
    train_Y = (train_Y == 'spam').astype(int)
    test_Y = (test_Y == 'spam').astype(int)
    
    return train_sequences, test_sequences, train_Y, test_Y, tokenizer, vocab_size

def create_model(vocab_size, max_len=100):
    """Create and compile the LSTM model"""
    print("\nCreating LSTM model...")
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
        tf.keras.layers.LSTM(16, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    print("Model created and compiled!")
    model.summary()
    
    return model

def train_model(model, train_sequences, train_Y, test_sequences, test_Y):
    """Train the model with callbacks"""
    print("\nTraining model...")
    
    # Callbacks
    es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True, verbose=1)
    lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=1)
    
    # Train
    history = model.fit(
        train_sequences, train_Y,
        validation_data=(test_sequences, test_Y),
        epochs=20,
        batch_size=32,
        callbacks=[lr, es],
        verbose=1
    )
    
    return history

def evaluate_model(model, test_sequences, test_Y):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    test_loss, test_accuracy = model.evaluate(test_sequences, test_Y, verbose=0)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    return test_loss, test_accuracy

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_model_and_tokenizer(model, tokenizer, model_dir='models'):
    """Save the trained model and tokenizer"""
    print(f"\nSaving model and tokenizer to '{model_dir}' directory...")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'spam_detection_model.h5')
    model.save(model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")

def main():
    """Main training pipeline"""
    print("=" * 50)
    print("EMAIL SPAM DETECTION - TRAINING PIPELINE")
    print("=" * 50)
    
    # Setup
    setup_nltk()
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Visualize initial distribution
    visualize_distribution(data, "Initial Email Label Distribution")
    
    # Balance dataset
    balanced_data = balance_dataset(data)
    
    # Visualize balanced distribution
    visualize_distribution(balanced_data, "Balanced Email Label Distribution")
    
    # Preprocess text
    processed_data = preprocess_text(balanced_data)
    
    # Generate word clouds
    print("\nGenerating word clouds...")
    plot_word_cloud(processed_data[processed_data['label'] == 'ham'], 'Non-Spam (Ham)')
    plot_word_cloud(processed_data[processed_data['label'] == 'spam'], 'Spam')
    
    # Prepare sequences
    max_len = 100
    train_seq, test_seq, train_Y, test_Y, tokenizer, vocab_size = prepare_sequences(processed_data, max_len)
    
    # Create model
    model = create_model(vocab_size, max_len)
    
    # Train model
    history = train_model(model, train_seq, train_Y, test_seq, test_Y)
    
    # Evaluate model
    evaluate_model(model, test_seq, test_Y)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Next steps:")
    print("1. Run 'python predict.py' to test predictions")
    print("2. Run 'python app.py' to start the web interface")

if __name__ == "__main__":
    main()
