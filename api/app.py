from flask import Flask, request, jsonify, render_template_string
import os
import sys
from predict import SpamDetector

app = Flask(__name__)

# Initialize spam detector globally
try:
    detector = SpamDetector()
    print("Spam detector initialized successfully!")
except Exception as e:
    print(f"Error initializing spam detector: {e}")
    detector = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Email Spam Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #555;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            box-sizing: border-box;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .result.spam {
            background-color: #ffebee;
            color: #c62828;
            border: 2px solid #ef5350;
        }
        .result.not-spam {
            background-color: #e8f5e8;
            color: #2e7d32;
            border: 2px solid #66bb6a;
        }
        .result.error {
            background-color: #fff8e1;
            color: #f57c00;
            border: 2px solid #ffb74d;
        }
        .confidence {
            font-size: 14px;
            margin-top: 10px;
            opacity: 0.8;
        }
        .loading {
            display: none;
            text-align: center;
            color: #666;
        }
        .samples {
            margin-top: 30px;
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 8px;
        }
        .sample-btn {
            background: #f0f0f0;
            border: 1px solid #ddd;
            padding: 8px 12px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            transition: background-color 0.2s;
        }
        .sample-btn:hover {
            background: #e0e0e0;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Email Spam Detection</h1>
        
        <form id="spamForm">
            <div class="form-group">
                <label for="email-text">Enter Email Text:</label>
                <textarea 
                    id="email-text" 
                    name="email" 
                    placeholder="Paste your email content here..."
                    required
                ></textarea>
            </div>
            
            <button type="submit" class="btn" id="predictBtn">
                🔍 Analyze Email
            </button>
        </form>
        
        <div class="loading" id="loading">
            <p>🔄 Analyzing email...</p>
        </div>
        
        <div id="result"></div>
        
        <div class="samples">
            <h3>📧 Try Sample Emails:</h3>
            <p style="font-size: 14px; color: #666; margin-bottom: 15px;">
                Click on any sample below to test the detector:
            </p>
            
            <button class="sample-btn" onclick="useSample('Congratulations! You have won $1,000,000! Click here immediately to claim your prize! No questions asked!')">
                🎉 Spam Sample 1
            </button>
            
            <button class="sample-btn" onclick="useSample('URGENT: Your account will be suspended! Verify your details now by clicking this link!')">
                ⚠️ Spam Sample 2
            </button>
            
            <button class="sample-btn" onclick="useSample('Hi John, can we schedule the team meeting for tomorrow at 2 PM? Please let me know if that works for you.')">
                📝 Ham Sample 1
            </button>
            
            <button class="sample-btn" onclick="useSample('Thank you for your presentation today. The quarterly report looks comprehensive and well-structured.')">
                👔 Ham Sample 2
            </button>
            
            <button class="sample-btn" onclick="useSample('FREE PILLS! VIAGRA 50% OFF! No prescription needed! Order now and get instant delivery!')">
                💊 Spam Sample 3
            </button>
            
            <button class="sample-btn" onclick="useSample('The project documentation has been updated. Please review the changes in the shared folder.')">
                📂 Ham Sample 3
            </button>
        </div>
        
        <div class="footer">
            <p>Built with TensorFlow & LSTM Neural Networks</p>
        </div>
    </div>
    
    <script>
        function useSample(text) {
            document.getElementById('email-text').value = text;
        }
        
        document.getElementById('spamForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const emailText = document.getElementById('email-text').value;
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            const predictBtn = document.getElementById('predictBtn');
            
            // Show loading
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            predictBtn.disabled = true;
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email: emailText })
                });
                
                const data = await response.json();
                
                // Hide loading
                loadingDiv.style.display = 'none';
                predictBtn.disabled = false;
                
                if (data.status === 'success') {
                    const isSpam = data.prediction === 'spam';
                    const icon = isSpam ? '🚨' : '✅';
                    const resultClass = isSpam ? 'spam' : 'not-spam';
                    
                    resultDiv.innerHTML = `
                        <div class="result ${resultClass}">
                            ${icon} ${data.prediction.toUpperCase()}
                            <div class="confidence">
                                Confidence: ${(data.probability * 100).toFixed(1)}%
                            </div>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            ⚠️ Error: ${data.message}
                        </div>
                    `;
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                predictBtn.disabled = false;
                
                resultDiv.innerHTML = `
                    <div class="result error">
                        ⚠️ Error: Failed to connect to server
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Main page with spam detection interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for spam prediction"""
    try:
        if detector is None:
            return jsonify({
                'status': 'error',
                'message': 'Spam detector not initialized. Please train the model first.'
            }), 500
        
        data = request.get_json()
        if not data or 'email' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Email text is required'
            }), 400
        
        email_text = data['email'].strip()
        if not email_text:
            return jsonify({
                'status': 'error',
                'message': 'Email text cannot be empty'
            }), 400
        
        # Make prediction
        label, probability, status = detector.predict_spam(email_text)
        
        if status == "Success":
            return jsonify({
                'status': 'success',
                'prediction': label,
                'probability': probability,
                'message': 'Prediction completed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': status
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_ready': detector is not None
    })

@app.route('/api/batch', methods=['POST'])
def predict_batch_api():
    """API endpoint for batch spam prediction"""
    try:
        if detector is None:
            return jsonify({
                'status': 'error',
                'message': 'Spam detector not initialized'
            }), 500
        
        data = request.get_json()
        if not data or 'emails' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Emails list is required'
            }), 400
        
        emails = data['emails']
        if not isinstance(emails, list):
            return jsonify({
                'status': 'error',
                'message': 'Emails must be a list'
            }), 400
        
        if len(emails) > 100:  # Limit batch size
            return jsonify({
                'status': 'error',
                'message': 'Maximum 100 emails per batch'
            }), 400
        
        # Make batch predictions
        results = detector.predict_batch(emails)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("EMAIL SPAM DETECTION WEB APPLICATION")
    print("=" * 60)
    
    if detector is None:
        print("⚠️  WARNING: Spam detector not initialized!")
        print("Please run 'python train_model.py' first to train the model.")
        print()
    else:
        print("✅ Spam detector ready!")
        print()
    
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/predict")
    print("🏥 Health check: http://localhost:5000/api/health")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
