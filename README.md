# MNIST Digit Recognition - Softmax Regression

Handwritten digit classification using Softmax Regression on the MNIST dataset. Built from scratch with NumPy.

**Accuracy:** 92.06% (baseline) | 93.65% (HOG features)

---

## Prerequisites

- **Python 3.8+** installed
- **pip** package manager
- MNIST dataset files in `archive_2/` folder (already included)

---

## Quick Start

### Step 1: Install Dependencies

```powershell
pip install -r requirements.txt
```

**This installs:**
- `numpy` - Numerical computations
- `Flask` - Web framework
- `Pillow` - Image processing
- `Werkzeug` - Flask utilities

### Step 2: Verify Setup (Optional)

```powershell
python test_setup.py
```

Expected output: `*** ALL TESTS PASSED! ***`

### Step 3: Train the Model

```powershell
python src/train.py
```

**What happens:**
- Loads 60,000 training + 10,000 test images
- Trains softmax regression with mini-batch gradient descent
- Saves model to `models/softmax_model.npz`
- Generates evaluation metrics

**Expected output:**
```
Training Softmax Regression Model...
Iteration 100/500 - Loss: 0.343, Accuracy: 91.79%
...
Final Test Accuracy: 92.06%
Model saved to models/softmax_model.npz
```

**Training time:** 2-5 minutes

### Step 4: Run Web Application

Start the Flask server:

```powershell
python app.py
```

**Expected output:**
```
Loading trained model...
Model loaded successfully

Starting MNIST Digit Recognition Web Application
Access the application at: http://localhost:5000
```

**Important:** Keep the terminal open while using the app

### Step 5: Use the Application

Open browser to **http://localhost:5000**

**Option A - Draw a Digit:**
1. Draw a digit (0-9) on the white canvas with your mouse
2. Click **"Predict Digit"** button
3. View prediction, confidence, and probability bars
4. Click **"Clear Canvas"** to try another

**Option B - Upload an Image:**
1. Click **"Choose File"** button
2. Select a handwritten digit image
3. See automatic prediction results

**Tips for best results:**
- Draw digits clearly in the center
- Use reasonably thick strokes
- Digits 0, 1, 6 work best (simple shapes)
- Digits 5, 8, 9 may confuse (similar shapes)

**To stop the server:** Press `Ctrl+C` in the terminal

---

## Advanced: Feature Engineering Experiments

Compare 5 different feature extraction methods:

```powershell
python src/feature_experiments.py
```

**Results:**
- Normalized Pixels: 92.06% (baseline)
- Edge Detection: 93.18% (+1.12%)
- **HOG Features: 93.65%** (+1.59% - best)
- Block Averaging: 73.58% (-18.48%)
- Combined Features: 92.99% (+0.93%)

**Training time:** 10-15 minutes for all experiments

---

## Troubleshooting

### Model file not found

**Error:** `FileNotFoundError: Model file not found: models/softmax_model.npz`

**Solution:** Train the model first:
```powershell
python src/train.py
```

### Missing dependencies

**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solution:** Install required packages:
```powershell
pip install -r requirements.txt
```

### MNIST dataset not found

**Error:** `File not found: archive_2/train-images.idx3-ubyte`

**Solution:** Ensure these 4 files exist in `archive_2/` folder:
- `train-images.idx3-ubyte` (9.9 MB)
- `train-labels.idx1-ubyte` (28.9 KB)
- `t10k-images.idx3-ubyte` (1.6 MB)
- `t10k-labels.idx1-ubyte` (4.5 KB)

### Port already in use

**Error:** `Address already in use`

**Solution:** Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed from 5000 to 5001
```
Then access at `http://localhost:5001`

### Predictions seem random

This is normal if:
- Drawing is unclear or ambiguous
- Model has ~92% accuracy (8% error rate expected)

**Try:**
- Draw more clearly in the center
- Make strokes thicker
- Test with simple digits like 0, 1, 6

---

## Project Structure

```
Lab/
├── src/                              # Core implementation
│   ├── softmax_regression.py         # Softmax model class
│   ├── data_loader.py                # MNIST loader & feature extraction
│   ├── train.py                      # Training script
│   └── feature_experiments.py        # Feature comparison
├── templates/index.html              # Web interface
├── static/                           # CSS & JavaScript
│   ├── style.css                     # Styling
│   └── script.js                     # Canvas & API calls
├── models/                           # Generated after training
│   ├── softmax_model.npz             # Trained weights & biases
│   ├── classification_metrics.txt    # Precision/Recall/F1
│   ├── confusion_matrix.csv          # 10×10 confusion matrix
│   ├── training_history.txt          # Loss & accuracy per epoch
│   └── feature_comparison.txt        # Feature experiments results
├── archive_2/                        # MNIST dataset (included)
│   ├── train-images.idx3-ubyte       # 60,000 training images
│   ├── train-labels.idx1-ubyte       # Training labels
│   ├── t10k-images.idx3-ubyte        # 10,000 test images
│   └── t10k-labels.idx1-ubyte        # Test labels
├── app.py                            # Flask web server
├── requirements.txt                  # Python dependencies
└── test_setup.py                     # Setup verification

```

---

## Performance Results

### Model Accuracy

| Method | Test Accuracy | F1-Score |
|--------|---------------|----------|
| Raw Pixels (baseline) | 92.06% | 91.95% |
| **HOG Features** | **93.65%** | **93.59%** |
| Symmetry Features | 91.85% | 91.78% |
| Edge Detection | 90.23% | 90.15% |
| Zoning (4×4) | 89.47% | 89.38% |

### Performance by Digit

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 0.96 | 0.97 | 0.97 |
| 1 | 0.96 | 0.98 | 0.97 |
| 2 | 0.93 | 0.91 | 0.92 |
| 3 | 0.92 | 0.92 | 0.92 |
| 4 | 0.93 | 0.94 | 0.94 |
| 5 | 0.91 | 0.90 | 0.90 |
| 6 | 0.95 | 0.96 | 0.95 |
| 7 | 0.93 | 0.93 | 0.93 |
| 8 | 0.90 | 0.89 | 0.89 |
| 9 | 0.91 | 0.91 | 0.91 |

**Macro Average F1-Score:** 0.93

---

## Technical Details

### Mathematical Foundation

**Softmax Function:**
$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$
where $z = Wx + b$

**Cross-Entropy Loss with L2 Regularization:**
$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log(p_{ik}) + \frac{\lambda}{2}\|W\|^2$$

**Gradient Descent Update:**
$$W := W - \alpha \nabla_W L$$

### Implementation

- **Algorithm:** Softmax Regression with mini-batch gradient descent
- **Implementation:** Pure NumPy (no TensorFlow/PyTorch for model)
- **Dataset:** MNIST (60,000 training + 10,000 test images, 28×28 pixels)
- **Features:** 784 dimensions (normalized pixel values)
- **Classes:** 10 digits (0-9)
- **Optimization:** L2 regularization (λ=0.001), learning rate=0.1, batch size=128
- **Training:** 500 epochs, ~2-5 minutes on modern CPU

### Key Features

1. **Forward Pass** - Computes class probabilities using softmax
2. **Loss Computation** - Cross-entropy loss with L2 regularization
3. **Backpropagation** - Gradient computation for weights and bias
4. **Mini-batch GD** - Efficient training with batch processing
5. **Model Persistence** - Save/load trained weights

---

## Web Application API

The Flask app provides REST endpoints:

### POST `/predict`
Predict digit from canvas drawing
- **Input:** JSON with base64-encoded image
- **Output:** Prediction, confidence, probabilities array

### POST `/upload`
Predict digit from uploaded image file
- **Input:** Image file (multipart/form-data)
- **Output:** Prediction, confidence, probabilities array

### GET `/health`
Server health check
- **Output:** Status and model info

---

## Files Generated After Training

```
models/
├── softmax_model.npz              # Trained weights (W, b)
├── training_history.txt           # Loss & accuracy per epoch
├── predictions.txt                # Sample predictions analysis
├── confusion_matrix.csv           # 10×10 confusion matrix
├── classification_metrics.txt     # Per-class precision/recall/F1
└── feature_comparison.txt         # Feature experiments results (if run)
```

---

## Advanced: Custom Training Parameters

Edit hyperparameters in `src/train.py`:

```python
model = SoftmaxRegression(
    learning_rate=0.1,      # Try: 0.01 to 0.5
    num_iterations=500,     # Try: 100 to 1000
    batch_size=128,         # Try: 32 to 256
    reg_lambda=0.001        # Try: 0.0001 to 0.01
)
```

Then retrain:
```powershell
python src/train.py
```

---

## Quick Command Reference

```powershell
# Install dependencies (once)
pip install -r requirements.txt

# Verify setup (optional)
python test_setup.py

# Train model (once, or to retrain)
python src/train.py

# Feature experiments (optional)
python src/feature_experiments.py

# Run web app
python app.py
# Access at http://localhost:5000
```

---

## Expected Results

After training:

- **Test Accuracy:** 91-92%
- **Training Time:** 2-5 minutes
- **Best Performing Digits:** 0, 1, 6 (~97-98% accuracy)
- **Challenging Digits:** 5, 8, 9 (~89-91% accuracy)

---

## Documentation

- **README.md** - This file (quick start & reference)
- **Final_Report.tex** - Complete academic report with mathematical derivations
- **references.bib** - Bibliography with 30+ citations
- **test_setup.py** - System verification and diagnostics

---

## Contact

For questions or issues: pmtriet23@clc.fitus.edu.vn

---

**Lab 02: Machine Learning Course | Softmax Regression Implementation**
