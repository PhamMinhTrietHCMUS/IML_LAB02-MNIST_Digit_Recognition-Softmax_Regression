"""
Flask Web Application for MNIST Digit Recognition
Allows users to draw digits and get predictions from the trained model
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from softmax_regression import SoftmaxRegression
from data_loader import preprocess_single_image

app = Flask(__name__)

# Global variable to store the model
model = None


def load_trained_model(model_path='models/softmax_model.npz'):
    """
    Load the trained softmax regression model
    """
    global model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = SoftmaxRegression()
    model.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    Receives a base64-encoded image and returns predictions
    """
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        preprocessed = preprocess_single_image(image, target_size=(28, 28))
        
        # Make prediction
        probabilities = model.predict_proba(preprocessed)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        # Prepare response with all class probabilities
        class_probabilities = {
            str(i): float(probabilities[i]) for i in range(10)
        }
        
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle file upload requests
    """
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the image
        image = Image.open(file.stream)
        
        # Preprocess the image
        preprocessed = preprocess_single_image(image, target_size=(28, 28))
        
        # Make prediction
        probabilities = model.predict_proba(preprocessed)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        # Prepare response with all class probabilities
        class_probabilities = {
            str(i): float(probabilities[i]) for i in range(10)
        }
        
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during file upload prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


def main():
    """
    Main function to run the Flask app
    """
    # Load the trained model
    print("Loading trained model...")
    try:
        load_trained_model('models/softmax_model.npz')
    except FileNotFoundError:
        print("\n" + "="*60)
        print("ERROR: Model file not found!")
        print("Please train the model first by running: python src/train.py")
        print("="*60 + "\n")
        sys.exit(1)
    
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting MNIST Digit Recognition Web Application")
    print("="*60)
    print("\nAccess the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
