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
from data_loader import preprocess_single_image, create_polynomial_features, extract_hog_features

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
    Render the main page with feature comparison
    """
    # Load feature comparison data
    features_data = []
    features_error = None
    best_feature = None
    
    try:
        # Feature configurations
        feature_configs = [
            {
                'name': 'Raw Pixel Features',
                'prefix': 'raw_',
                'dimensions': 784,
                'icon': '🔲',
                'description': 'Direct pixel intensity values (28×28=784). Simplest baseline using normalized pixel values without any transformation.'
            },
            {
                'name': 'Polynomial Features',
                'prefix': 'poly_',
                'dimensions': 1568,
                'icon': '📐',
                'description': 'Original pixels + squared terms (784×2=1568). Captures non-linear relationships: [x₁, x₂, ..., x₇₈₄, x₁², x₂², ..., x₇₈₄²]'
            },
            {
                'name': 'HOG Features',
                'prefix': 'hog_',
                'dimensions': 784,
                'icon': '🎨',
                'description': 'Histogram of Oriented Gradients (784). Gradient magnitude features that capture edge and contour information.'
            }
        ]
        
        best_accuracy = 0
        best_idx = 0
        
        for idx, config in enumerate(feature_configs):
            try:
                # Read metrics file
                metrics_file = f"models/{config['prefix']}classification_metrics.txt"
                
                if not os.path.exists(metrics_file):
                    continue
                
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse accuracy
                accuracy_line = lines[0].strip()
                accuracy = float(accuracy_line.split(':')[1].strip().replace('%', ''))
                
                # Parse average metrics (last 3 lines)
                avg_precision = float(lines[-3].split(':')[1].strip())
                avg_recall = float(lines[-2].split(':')[1].strip())
                avg_f1 = float(lines[-1].split(':')[1].strip())
                
                features_data.append({
                    'name': config['name'],
                    'dimensions': config['dimensions'],
                    'icon': config['icon'],
                    'description': config['description'],
                    'accuracy': accuracy,
                    'avg_precision': avg_precision,
                    'avg_recall': avg_recall,
                    'avg_f1': avg_f1,
                    'is_best': False
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_idx = len(features_data) - 1
                    
            except Exception as e:
                print(f"Error loading {config['name']}: {e}")
                continue
        
        if features_data:
            # Mark best feature
            features_data[best_idx]['is_best'] = True
            # Sort by accuracy (descending)
            features_data.sort(key=lambda x: x['accuracy'], reverse=True)
            best_feature = features_data[0]
        else:
            features_error = "No comparison results found. Training may still be in progress..."
            
    except Exception as e:
        features_error = f"Error loading feature comparison: {str(e)}"
        print(features_error)
    
    return render_template('index.html', 
                         features=features_data,
                         features_error=features_error,
                         best_feature=best_feature)


@app.route('/results')
def results():
    """
    Display feature comparison results
    """
    try:
        # Load results from files
        features_data = []
        
        # Feature configurations
        feature_configs = [
            {
                'name': 'Raw Pixel Features',
                'prefix': 'raw_',
                'dimensions': 784,
                'icon': '🔲',
                'description': 'Direct pixel intensity values normalized to [0, 1]. This is the simplest feature representation, using each pixel as a separate feature without any transformation.'
            },
            {
                'name': 'Polynomial Features',
                'prefix': 'poly_',
                'dimensions': 1568,
                'icon': '📐',
                'description': 'Enhanced feature set created by adding squared terms of each pixel value. Formula: [x₁, x₂, ..., x₇₈₄, x₁², x₂², ..., x₇₈₄²]. This captures non-linear relationships between pixel intensities.'
            },
            {
                'name': 'HOG Features',
                'prefix': 'hog_',
                'dimensions': 784,
                'icon': '🎨',
                'description': 'Histogram of Oriented Gradients - extracts gradient magnitude at each pixel location. Computes image gradients using finite differences and uses magnitude as features. Effective at capturing edge and contour information.'
            }
        ]
        
        best_accuracy = 0
        best_idx = 0
        
        for idx, config in enumerate(feature_configs):
            try:
                # Read metrics file
                metrics_file = f"models/{config['prefix']}classification_metrics.txt"
                
                if not os.path.exists(metrics_file):
                    continue
                
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse accuracy
                accuracy_line = lines[0].strip()
                accuracy = float(accuracy_line.split(':')[1].strip().replace('%', ''))
                
                # Parse average metrics (last 3 lines)
                avg_precision = float(lines[-3].split(':')[1].strip())
                avg_recall = float(lines[-2].split(':')[1].strip())
                avg_f1 = float(lines[-1].split(':')[1].strip())
                
                # Load confusion matrix
                cm_file = f"models/{config['prefix']}confusion_matrix.csv"
                confusion_matrix = []
                if os.path.exists(cm_file):
                    confusion_matrix = np.loadtxt(cm_file, delimiter=',', dtype=int).tolist()
                
                features_data.append({
                    'name': config['name'],
                    'dimensions': config['dimensions'],
                    'icon': config['icon'],
                    'description': config['description'],
                    'accuracy': accuracy,
                    'avg_precision': avg_precision,
                    'avg_recall': avg_recall,
                    'avg_f1': avg_f1,
                    'confusion_matrix': confusion_matrix,
                    'is_best': False
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_idx = len(features_data) - 1
                    
            except Exception as e:
                print(f"Error loading {config['name']}: {e}")
                continue
        
        if not features_data:
            return render_template('results.html', error="No comparison results found. Please run: python src/compare_features.py")
        
        # Mark best feature
        features_data[best_idx]['is_best'] = True
        
        # Sort by accuracy (descending)
        features_data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return render_template('results.html', 
                             features=features_data,
                             best_feature=features_data[0],
                             error=None)
        
    except Exception as e:
        return render_template('results.html', error=f"Error loading results: {str(e)}")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from all 3 models
    Receives a base64-encoded image and returns predictions from Raw, Poly, and HOG models
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
        
        # Preprocess the image (raw features)
        preprocessed_raw = preprocess_single_image(image, target_size=(28, 28))
        
        # Prepare results for all models
        all_predictions = []
        
        # For HOG: need unnormalized image (0-255 range)
        img_unnormalized = (preprocessed_raw * 255.0).reshape(28, 28)
        
        # Model configurations
        model_configs = [
            {
                'name': 'Raw Pixels',
                'model_path': 'models/softmax_model_raw.npz',
                'icon': '🔲',
                'features': preprocessed_raw,
                'color': '#3498db'
            },
            {
                'name': 'Polynomial',
                'model_path': 'models/softmax_model_poly.npz',
                'icon': '📐',
                'features': create_polynomial_features(preprocessed_raw, degree=2),
                'color': '#2ecc71'
            },
            {
                'name': 'HOG',
                'model_path': 'models/softmax_model_hog.npz',
                'icon': '🎨',
                'features': extract_hog_features(img_unnormalized.reshape(1, 28, 28)).reshape(1, -1) / 255.0,
                'color': '#e74c3c'
            }
        ]
        
        # Try to get predictions from each model
        for config in model_configs:
            if os.path.exists(config['model_path']):
                try:
                    # Load model
                    temp_model = SoftmaxRegression()
                    temp_model.load_model(config['model_path'])
                    
                    # Make prediction
                    probabilities = temp_model.predict_proba(config['features'])[0]
                    prediction = int(np.argmax(probabilities))
                    confidence = float(probabilities[prediction])
                    
                    all_predictions.append({
                        'name': config['name'],
                        'icon': config['icon'],
                        'prediction': prediction,
                        'confidence': confidence,
                        'probabilities': {str(i): float(probabilities[i]) for i in range(10)},
                        'color': config['color']
                    })
                except Exception as e:
                    print(f"Error loading {config['name']} model: {e}")
                    continue
        
        # If no models available, use default model (polynomial)
        if not all_predictions:
            preprocessed = create_polynomial_features(preprocessed_raw, degree=2)
            probabilities = model.predict_proba(preprocessed)[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            
            all_predictions.append({
                'name': 'Default (Polynomial)',
                'icon': '📐',
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {str(i): float(probabilities[i]) for i in range(10)},
                'color': '#2ecc71'
            })
        
        response = {
            'success': True,
            'models': all_predictions,
            'prediction': all_predictions[0]['prediction'] if all_predictions else 0,
            'confidence': all_predictions[0]['confidence'] if all_predictions else 0,
            'probabilities': all_predictions[0]['probabilities'] if all_predictions else {}
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle file upload requests with predictions from all 3 models
    """
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Read the image
        image = Image.open(file.stream)
        
        # Preprocess the image (raw features)
        preprocessed_raw = preprocess_single_image(image, target_size=(28, 28))
        
        # Prepare results for all models
        all_predictions = []
        
        # For HOG: need unnormalized image (0-255 range)
        img_unnormalized = (preprocessed_raw * 255.0).reshape(28, 28)
        
        # Model configurations
        model_configs = [
            {
                'name': 'Raw Pixels',
                'model_path': 'models/softmax_model_raw.npz',
                'icon': '🔲',
                'features': preprocessed_raw,
                'color': '#3498db'
            },
            {
                'name': 'Polynomial',
                'model_path': 'models/softmax_model_poly.npz',
                'icon': '📐',
                'features': create_polynomial_features(preprocessed_raw, degree=2),
                'color': '#2ecc71'
            },
            {
                'name': 'HOG',
                'model_path': 'models/softmax_model_hog.npz',
                'icon': '🎨',
                'features': extract_hog_features(img_unnormalized.reshape(1, 28, 28)).reshape(1, -1) / 255.0,
                'color': '#e74c3c'
            }
        ]
        
        # Try to get predictions from each model
        for config in model_configs:
            if os.path.exists(config['model_path']):
                try:
                    # Load model
                    temp_model = SoftmaxRegression()
                    temp_model.load_model(config['model_path'])
                    
                    # Make prediction
                    probabilities = temp_model.predict_proba(config['features'])[0]
                    prediction = int(np.argmax(probabilities))
                    confidence = float(probabilities[prediction])
                    
                    all_predictions.append({
                        'name': config['name'],
                        'icon': config['icon'],
                        'prediction': prediction,
                        'confidence': confidence,
                        'probabilities': {str(i): float(probabilities[i]) for i in range(10)},
                        'color': config['color']
                    })
                except Exception as e:
                    print(f"Error loading {config['name']} model: {e}")
                    continue
        
        # If no models available, use default model
        if not all_predictions:
            preprocessed = create_polynomial_features(preprocessed_raw, degree=2)
            probabilities = model.predict_proba(preprocessed)[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            
            all_predictions.append({
                'name': 'Default (Polynomial)',
                'icon': '📐',
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {str(i): float(probabilities[i]) for i in range(10)},
                'color': '#2ecc71'
            })
        
        response = {
            'success': True,
            'models': all_predictions,
            'prediction': all_predictions[0]['prediction'] if all_predictions else 0,
            'confidence': all_predictions[0]['confidence'] if all_predictions else 0,
            'probabilities': all_predictions[0]['probabilities'] if all_predictions else {}
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
