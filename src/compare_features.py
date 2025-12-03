"""
Compare performance of different feature designs
Compares Raw Pixels, Polynomial Features, and HOG Features
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_mnist_data,
    preprocess_images,
    split_train_val,
    create_polynomial_features,
    extract_hog_features
)
from softmax_regression import SoftmaxRegression
from evaluation import evaluate_model, print_evaluation_results, save_results


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_name, 
                       learning_rate=0.001, optimizer='adam'):
    """Train and evaluate a model"""
    print(f"\n{'='*60}")
    print(f"Training with {feature_name}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"{'='*60}")
    
    # Split validation
    X_train_split, X_val, y_train_split, y_val = split_train_val(
        X_train, y_train, val_ratio=0.1
    )
    
    # Train
    model = SoftmaxRegression(
        learning_rate=learning_rate,
        num_iterations=500,
        batch_size=128,
        reg_lambda=0.001,
        optimizer=optimizer
    )
    
    model.fit(X_train_split, y_train_split, X_val, y_val)
    
    # Evaluate
    accuracy, metrics = evaluate_model(model, X_test, y_test)
    print_evaluation_results(accuracy, metrics)
    
    return accuracy, metrics, model


def main():
    print("="*70)
    print("FEATURE DESIGN COMPARISON - MNIST SOFTMAX REGRESSION")
    print("="*70)
    
    # Load data
    X_train_raw, y_train, X_test_raw, y_test = load_mnist_data()
    
    results = {}
    models = {}
    
    # 1. Raw Pixel Features
    print("\n\n" + "="*70)
    print("EXPERIMENT 1: RAW PIXEL FEATURES")
    print("="*70)
    print("Description: Direct pixel intensity values (28x28 = 784 features)")
    print("Preprocessing: Normalize to [0, 1] and flatten")
    
    X_train_raw_flat = preprocess_images(X_train_raw)
    X_test_raw_flat = preprocess_images(X_test_raw)
    
    acc1, metrics1, model1 = train_and_evaluate(
        X_train_raw_flat, y_train, 
        X_test_raw_flat, y_test,
        "Raw Pixel Features",
        learning_rate=0.001,
        optimizer='adam'
    )
    results['Raw Pixels (784)'] = acc1
    models['raw'] = (model1, X_test_raw_flat, y_test)
    
    # Save results
    model1.save_model('models/softmax_model_raw.npz')
    save_results(model1, X_test_raw_flat, y_test, prefix='models/raw_')
    
    # 2. Polynomial Features
    print("\n\n" + "="*70)
    print("EXPERIMENT 2: POLYNOMIAL FEATURES (Degree 2)")
    print("="*70)
    print("Description: Add squared terms to capture non-linear relationships")
    print("Formula: [x1, x2, ..., x784, x1², x2², ..., x784²]")
    print("Feature dimensions: 784 → 1568")
    
    X_train_poly = create_polynomial_features(X_train_raw_flat, degree=2)
    X_test_poly = create_polynomial_features(X_test_raw_flat, degree=2)
    
    acc2, metrics2, model2 = train_and_evaluate(
        X_train_poly, y_train,
        X_test_poly, y_test,
        "Polynomial Features (degree=2)",
        learning_rate=0.001,
        optimizer='adam'
    )
    results['Polynomial (1568)'] = acc2
    models['poly'] = (model2, X_test_poly, y_test)
    
    # Save results
    model2.save_model('models/softmax_model_poly.npz')
    save_results(model2, X_test_poly, y_test, prefix='models/poly_')
    
    # 3. HOG Features
    print("\n\n" + "="*70)
    print("EXPERIMENT 3: HOG (Histogram of Oriented Gradients) FEATURES")
    print("="*70)
    print("Description: Gradient-based features capturing edge orientations")
    print("Method: Compute image gradients (dx, dy) and their magnitudes")
    print("Feature dimensions: 784 (gradient magnitude per pixel)")
    
    X_train_hog = extract_hog_features(X_train_raw) / 255.0
    X_test_hog = extract_hog_features(X_test_raw) / 255.0
    
    acc3, metrics3, model3 = train_and_evaluate(
        X_train_hog, y_train,
        X_test_hog, y_test,
        "HOG Features",
        learning_rate=0.001,
        optimizer='adam'
    )
    results['HOG (784)'] = acc3
    models['hog'] = (model3, X_test_hog, y_test)
    
    # Save results
    model3.save_model('models/softmax_model_hog.npz')
    save_results(model3, X_test_hog, y_test, prefix='models/hog_')
    
    # Summary
    print("\n\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Feature Design':<35} {'Dimensions':<15} {'Accuracy':<15}")
    print("-" * 70)
    
    feature_details = {
        'Raw Pixels (784)': 784,
        'Polynomial (1568)': 1568,
        'HOG (784)': 784
    }
    
    for feature_name, accuracy in results.items():
        dims = feature_details[feature_name]
        print(f"{feature_name:<35} {dims:<15} {accuracy:>6.2f}%")
    
    # Find best
    best_feature = max(results, key=results.get)
    best_accuracy = results[best_feature]
    
    print("\n" + "-" * 70)
    print(f"BEST PERFORMING: {best_feature} with {best_accuracy:.2f}% accuracy")
    print("="*70)
    
    # Save comparison to file
    with open('models/feature_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FEATURE DESIGN COMPARISON - MNIST SOFTMAX REGRESSION\n")
        f.write("="*70 + "\n\n")
        
        f.write("EXPERIMENT DETAILS:\n\n")
        
        f.write("1. RAW PIXEL FEATURES (784 dimensions)\n")
        f.write("   - Direct pixel intensity values\n")
        f.write("   - Normalized to [0, 1]\n")
        f.write(f"   - Accuracy: {results['Raw Pixels (784)']:.2f}%\n\n")
        
        f.write("2. POLYNOMIAL FEATURES (1568 dimensions)\n")
        f.write("   - Original pixels + squared terms\n")
        f.write("   - Captures non-linear relationships\n")
        f.write(f"   - Accuracy: {results['Polynomial (1568)']:.2f}%\n\n")
        
        f.write("3. HOG FEATURES (784 dimensions)\n")
        f.write("   - Gradient magnitude features\n")
        f.write("   - Captures edge information\n")
        f.write(f"   - Accuracy: {results['HOG (784)']:.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write(f"BEST: {best_feature} - {best_accuracy:.2f}%\n")
        f.write("="*70 + "\n")
    
    print("\nComparison results saved to 'models/feature_comparison.txt'")
    
    return results


if __name__ == '__main__':
    results = main()
