# Email Spam Detection Project

## Overview
This project implements an email spam detection system using TensorFlow/Keras with an LSTM neural network. The model classifies emails as spam or non-spam (ham) with high accuracy.

## Features
- Data preprocessing and cleaning
- Visualizations including word clouds and label distributions
- LSTM-based deep learning model for classification
- Interactive CLI prediction tool
- Flask web interface & REST API
- Batch email processing support
- Automated testing suite
- One-step automated setup script

## Installation Instructions
1. Clone or download the project.
2. Create a Python virtual environment (recommended).
3. Run `python setup.py` for one-click setup or follow manual steps:
   - Install required packages: `pip install -r requirements.txt`
   - Download NLTK stopwords: Run `python -c "import nltk; nltk.download('stopwords')"`

## Running the Project

### Training the Model
