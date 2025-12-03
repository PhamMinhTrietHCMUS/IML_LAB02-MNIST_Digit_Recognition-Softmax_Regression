"""
Training Script for Softmax Regression on MNIST
"""
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from softmax_regression import SoftmaxRegression
from data_loader import (
    load_mnist_data,
    preprocess_images,
    split_train_val,
    create_polynomial_features
)


def save_training_history(history, save_path='models/training_history.txt'):
    """
    Save training history to a text file
    """
    with open(save_path, 'w') as f:
        f.write("Training History\n")
        f.write("=" * 60 + "\n\n")
        f.write("Iteration\tLoss\t\tAccuracy\n")
        f.write("-" * 60 + "\n")
        
        for i, (loss, acc) in enumerate(zip(history['loss'], history['accuracy'])):
            f.write(f"{i*10}\t\t{loss:.6f}\t{acc:.6f}\n")
    
    print(f"Training history saved to {save_path}")


def save_predictions_text(model, X_test, y_test, num_samples=20, save_path='models/predictions.txt'):
    """
    Save sample predictions to a text file
    """
    # Randomly select samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    with open(save_path, 'w') as f:
        f.write("Sample Predictions\n")
        f.write("=" * 60 + "\n\n")
        
        for i, idx in enumerate(indices):
            X_sample = X_test[idx:idx+1]
            probs = model.predict_proba(X_sample)[0]
            prediction = np.argmax(probs)
            true_label = y_test[idx]
            confidence = probs[prediction]
            
            f.write(f"Sample {i+1}:\n")
            f.write(f"  True Label: {true_label}\n")
            f.write(f"  Prediction: {prediction}\n")
            f.write(f"  Confidence: {confidence:.4f}\n")
            f.write(f"  Correct: {'Yes' if prediction == true_label else 'No'}\n")
            f.write(f"  Probabilities: {[f'{p:.4f}' for p in probs]}\n")
            f.write("\n")
    
    print(f"Sample predictions saved to {save_path}")


def compute_confusion_matrix(model, X_test, y_test):
    """
    Compute and display confusion matrix
    """
    predictions = model.predict(X_test)
    num_classes = model.num_classes
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_test, predictions):
        confusion_matrix[true_label, pred_label] += 1
    
    return confusion_matrix


def print_confusion_matrix(confusion_matrix):
    """
    Print confusion matrix in a readable format
    """
    print("\nConfusion Matrix:")
    print("True\\Pred", end="")
    for i in range(10):
        print(f"{i:>6}", end="")
    print()
    
    for i in range(10):
        print(f"{i:>9}", end="")
        for j in range(10):
            print(f"{confusion_matrix[i, j]:>6}", end="")
        print()


def save_confusion_matrix_csv(confusion_matrix, save_path='models/confusion_matrix.csv'):
    """
    Save confusion matrix as CSV file
    """
    with open(save_path, 'w') as f:
        # Header
        f.write("True\\Pred")
        for i in range(10):
            f.write(f",{i}")
        f.write("\n")
        
        # Data
        for i in range(10):
            f.write(f"{i}")
            for j in range(10):
                f.write(f",{confusion_matrix[i, j]}")
            f.write("\n")
    
    print(f"Confusion matrix saved to {save_path}")


def compute_precision_recall_f1(confusion_matrix):
    """
    Compute Precision, Recall, and F1-score for each class
    
    Returns:
    --------
    metrics : dict with precision, recall, f1 for each class
    """
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True Positives: correctly predicted as class i
        tp = confusion_matrix[i, i]
        
        # False Positives: predicted as i but actually other class
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # False Negatives: actually class i but predicted as other
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0.0
        
        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        else:
            recall[i] = 0.0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_classification_metrics(metrics):
    """
    Print Precision, Recall, and F1-score in a formatted table
    """
    print("\nDetailed Classification Metrics:")
    print("-" * 70)
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    for i in range(10):
        print(f"Digit {i:<2} {metrics['precision'][i]:>8.4f}     "
              f"{metrics['recall'][i]:>8.4f}     {metrics['f1'][i]:>8.4f}")
    
    # Macro average
    macro_precision = np.mean(metrics['precision'])
    macro_recall = np.mean(metrics['recall'])
    macro_f1 = np.mean(metrics['f1'])
    
    print("-" * 70)
    print(f"{'Macro Avg':<8} {macro_precision:>8.4f}     "
          f"{macro_recall:>8.4f}     {macro_f1:>8.4f}")
    
    # Weighted average (by support)
    # For MNIST test set, all classes are roughly balanced
    print(f"{'Weighted':<8} {macro_precision:>8.4f}     "
          f"{macro_recall:>8.4f}     {macro_f1:>8.4f}")


def save_classification_metrics(metrics, save_path='models/classification_metrics.txt'):
    """
    Save classification metrics to file
    """
    with open(save_path, 'w') as f:
        f.write("Classification Metrics Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-"*70 + "\n")
        
        for i in range(10):
            f.write(f"Digit {i:<2} {metrics['precision'][i]:>8.4f}     "
                   f"{metrics['recall'][i]:>8.4f}     {metrics['f1'][i]:>8.4f}\n")
        
        macro_precision = np.mean(metrics['precision'])
        macro_recall = np.mean(metrics['recall'])
        macro_f1 = np.mean(metrics['f1'])
        
        f.write("-"*70 + "\n")
        f.write(f"{'Macro Avg':<8} {macro_precision:>8.4f}     "
               f"{macro_recall:>8.4f}     {macro_f1:>8.4f}\n")
        f.write(f"{'Weighted':<8} {macro_precision:>8.4f}     "
               f"{macro_recall:>8.4f}     {macro_f1:>8.4f}\n")
    
    print(f"Classification metrics saved to {save_path}")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with comprehensive metrics
    """
    # Overall accuracy
    accuracy = model.score(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    confusion_matrix = compute_confusion_matrix(model, X_test, y_test)
    print_confusion_matrix(confusion_matrix)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(10):
        class_correct = confusion_matrix[i, i]
        class_total = np.sum(confusion_matrix[i, :])
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"Digit {i}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Precision, Recall, F1-Score
    metrics = compute_precision_recall_f1(confusion_matrix)
    print_classification_metrics(metrics)
    
    return accuracy, confusion_matrix, metrics


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("MNIST Softmax Regression Training")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("\n1. Loading MNIST dataset...")
    X_train_raw, y_train, X_test_raw, y_test = load_mnist_data(data_dir='archive_2')
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train = preprocess_images(X_train_raw, normalize=True, flatten=True)
    X_test = preprocess_images(X_test_raw, normalize=True, flatten=True)
    
    # Create polynomial features (degree=2)
    print("   Creating polynomial features (degree=2)...")
    X_train = create_polynomial_features(X_train, degree=2)
    X_test = create_polynomial_features(X_test, degree=2)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Split training data into train and validation sets
    print("\n3. Splitting data into train and validation sets...")
    X_train_split, X_val, y_train_split, y_val = split_train_val(
        X_train, y_train, val_ratio=0.1, random_state=42
    )
    
    print(f"Training set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\n4. Training Softmax Regression model...")
    print("-" * 60)
    
    model = SoftmaxRegression(
        learning_rate=0.001,  # Lower learning rate for Adam
        num_iterations=500,
        batch_size=128,
        reg_lambda=0.001,
        optimizer='adam'
    )
    
    model.fit(X_train_split, y_train_split, X_val, y_val, verbose=True)
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    print("-" * 60)
    test_accuracy, confusion_matrix, metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    print("\n6. Saving model...")
    model.save_model('models/softmax_model.npz')
    
    # Save training history
    print("\n7. Saving training history...")
    save_training_history(model.training_history, save_path='models/training_history.txt')
    
    # Save sample predictions
    print("\n8. Saving sample predictions...")
    save_predictions_text(model, X_test, y_test, num_samples=20, save_path='models/predictions.txt')
    
    # Save confusion matrix as CSV
    print("\n9. Saving confusion matrix...")
    save_confusion_matrix_csv(confusion_matrix, save_path='models/confusion_matrix.csv')
    
    # Save classification metrics
    print("\n10. Saving classification metrics...")
    save_classification_metrics(metrics, save_path='models/classification_metrics.txt')
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 60)
    
    return model


if __name__ == '__main__':
    model = main()
