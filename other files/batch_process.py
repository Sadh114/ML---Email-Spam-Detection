import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from predict import SpamDetector
import json

def process_csv_file(input_file, output_file=None, detector=None):
    """Process a CSV file containing emails and add spam predictions"""
    
    if detector is None:
        detector = SpamDetector()
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} emails from {input_file}")
        
        # Check if required columns exist
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column with email content")
        
        # Add prediction columns
        predictions = []
        probabilities = []
        statuses = []
        
        print("Processing emails...")
        for i, email_text in enumerate(df['text']):
            if i % 100 == 0:
                print(f"Processed {i}/{len(df)} emails...")
            
            label, prob, status = detector.predict_spam(str(email_text))
            predictions.append(label)
            probabilities.append(prob)
            statuses.append(status)
        
        # Add results to dataframe
        df['predicted_label'] = predictions
        df['spam_probability'] = probabilities
        df['prediction_status'] = statuses
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_predictions.csv"
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print summary statistics
        print_summary_statistics(df)
        
        return df
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def process_text_file(input_file, output_file=None, detector=None):
    """Process a text file where each line is an email"""
    
    if detector is None:
        detector = SpamDetector()
    
    try:
        # Read the text file
        with open(input_file, 'r', encoding='utf-8') as f:
            emails = f.readlines()
        
        emails = [email.strip() for email in emails if email.strip()]
        print(f"Loaded {len(emails)} emails from {input_file}")
        
        # Process emails
        results = []
        print("Processing emails...")
        
        for i, email_text in enumerate(emails):
            if i % 100 == 0:
                print(f"Processed {i}/{len(emails)} emails...")
            
            label, prob, status = detector.predict_spam(email_text)
            results.append({
                'email_index': i + 1,
                'email_text': email_text,
                'predicted_label': label,
                'spam_probability': prob,
                'prediction_status': status
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_predictions.csv"
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print summary statistics
        print_summary_statistics(df)
        
        return df
        
    except Exception as e:
        print(f"Error processing text file: {e}")
        return None

def print_summary_statistics(df):
    """Print summary statistics of predictions"""
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    
    total_emails = len(df)
    spam_count = len(df[df['predicted_label'] == 'spam'])
    ham_count = len(df[df['predicted_label'] == 'not spam'])
    error_count = len(df[df['prediction_status'] != 'Success'])
    
    print(f"Total emails processed: {total_emails}")
    print(f"Predicted as SPAM: {spam_count} ({spam_count/total_emails*100:.1f}%)")
    print(f"Predicted as HAM: {ham_count} ({ham_count/total_emails*100:.1f}%)")
    print(f"Processing errors: {error_count}")
    
    if spam_count > 0:
        spam_probs = df[df['predicted_label'] == 'spam']['spam_probability']
        print(f"\nSpam probability statistics:")
        print(f"  Average: {spam_probs.mean():.3f}")
        print(f"  Median: {spam_probs.median():.3f}")
        print(f"  Min: {spam_probs.min():.3f}")
        print(f"  Max: {spam_probs.max():.3f}")
    
    if ham_count > 0:
        ham_probs = df[df['predicted_label'] == 'not spam']['spam_probability']
        print(f"\nHam probability statistics:")
        print(f"  Average: {ham_probs.mean():.3f}")
        print(f"  Median: {ham_probs.median():.3f}")
        print(f"  Min: {ham_probs.min():.3f}")
        print(f"  Max: {ham_probs.max():.3f}")

def process_json_file(input_file, output_file=None, detector=None):
    """Process a JSON file containing emails"""
    
    if detector is None:
        detector = SpamDetector()
    
    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            emails = data
        elif isinstance(data, dict):
            if 'emails' in data:
                emails = data['emails']
            else:
                emails = [str(v) for v in data.values()]
        else:
            raise ValueError("Unsupported JSON structure")
        
        print(f"Loaded {len(emails)} emails from {input_file}")
        
        # Process emails
        results = []
        print("Processing emails...")
        
        for i, email_text in enumerate(emails):
            if i % 100 == 0:
                print(f"Processed {i}/{len(emails)} emails...")
            
            label, prob, status = detector.predict_spam(str(email_text))
            results.append({
                'email_index': i + 1,
                'email_text': str(email_text),
                'predicted_label': label,
                'spam_probability': prob,
                'prediction_status': status
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_predictions.csv"
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print summary statistics
        print_summary_statistics(df)
        
        return df
        
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Batch process emails for spam detection')
    parser.add_argument('input_file', help='Input file path (CSV, TXT, or JSON)')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-t', '--type', choices=['csv', 'txt', 'json'], 
                       help='File type (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    print("="*60)
    print("EMAIL SPAM DETECTION - BATCH PROCESSING")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Initialize detector
        print("Loading spam detection model...")
        detector = SpamDetector()
        
        # Determine file type
        if args.type:
            file_type = args.type
        else:
            file_ext = os.path.splitext(args.input_file)[1].lower()
            if file_ext == '.csv':
                file_type = 'csv'
            elif file_ext == '.txt':
                file_type = 'txt'
            elif file_ext == '.json':
                file_type = 'json'
            else:
                print(f"Unknown file extension: {file_ext}")
                print("Please specify file type with --type argument")
                return
        
        # Process file based on type
        start_time = datetime.now()
        
        if file_type == 'csv':
            result = process_csv_file(args.input_file, args.output, detector)
        elif file_type == 'txt':
            result = process_text_file(args.input_file, args.output, detector)
        elif file_type == 'json':
            result = process_json_file(args.input_file, args.output, detector)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result is not None:
            print(f"\nProcessing completed in {processing_time:.2f} seconds")
            print(f"Average time per email: {processing_time/len(result):.3f} seconds")
        else:
            print("Processing failed!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments provided
        print("="*60)
        print("EMAIL SPAM DETECTION - BATCH PROCESSING")
        print("="*60)
        
        input_file = input("Enter input file path: ").strip()
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            sys.exit(1)
        
        output_file = input("Enter output file path (press Enter for auto): ").strip()
        if not output_file:
            output_file = None
            
        try:
            detector = SpamDetector()
            
            file_ext = os.path.splitext(input_file)[1].lower()
            if file_ext == '.csv':
                process_csv_file(input_file, output_file, detector)
            elif file_ext == '.txt':
                process_text_file(input_file, output_file, detector)
            elif file_ext == '.json':
                process_json_file(input_file, output_file, detector)
            else:
                print(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        main()