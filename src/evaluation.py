"""
Model Evaluation Utilities
Provides functions for evaluating classification performance
"""
import numpy as np


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Compute confusion matrix
    
    Parameters:
    -----------
    y_true : numpy array
        True labels
    y_pred : numpy array
        Predicted labels
    num_classes : int
        Number of classes
        
    Returns:
    --------
    confusion_matrix : numpy array of shape (num_classes, num_classes)
        Confusion matrix
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def compute_metrics_per_class(confusion_matrix):
    """
    Compute precision, recall, and F1-score for each class
    
    Parameters:
    -----------
    confusion_matrix : numpy array
        Confusion matrix
        
    Returns:
    --------
    metrics : dict
        Dictionary containing precision, recall, and f1-score for each class
    """
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True Positives
        tp = confusion_matrix[i, i]
        
        # False Positives (predicted as i but actually not i)
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # False Negatives (actually i but predicted as something else)
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Precision
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score
        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_score[i] = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return metrics
    
    Parameters:
    -----------
    model : SoftmaxRegression
        Trained model
    X_test : numpy array
        Test features
    y_test : numpy array
        Test labels
        
    Returns:
    --------
    accuracy : float
        Overall accuracy
    metrics : dict
        Dictionary containing confusion matrix and per-class metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(y_test, y_pred)
    
    # Compute per-class metrics
    class_metrics = compute_metrics_per_class(cm)
    
    return accuracy, {
        'confusion_matrix': cm,
        'precision': class_metrics['precision'],
        'recall': class_metrics['recall'],
        'f1_score': class_metrics['f1_score']
    }


def print_evaluation_results(accuracy, metrics):
    """
    Print evaluation results in a formatted manner
    
    Parameters:
    -----------
    accuracy : float
        Overall accuracy
    metrics : dict
        Dictionary containing metrics
    """
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print("\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    for i in range(len(metrics['precision'])):
        print(f"{i:<10} {metrics['precision'][i]:<12.4f} {metrics['recall'][i]:<12.4f} {metrics['f1_score'][i]:<12.4f}")
    
    print(f"\nMacro Average:")
    print(f"Precision: {np.mean(metrics['precision']):.4f}")
    print(f"Recall: {np.mean(metrics['recall']):.4f}")
    print(f"F1-Score: {np.mean(metrics['f1_score']):.4f}")


def save_results(model, X_test, y_test, prefix='models/'):
    """
    Save evaluation results to files
    
    Parameters:
    -----------
    model : SoftmaxRegression
        Trained model
    X_test : numpy array
        Test features
    y_test : numpy array
        Test labels
    prefix : str
        Prefix for output files
    """
    accuracy, metrics = evaluate_model(model, X_test, y_test)
    
    # Save confusion matrix
    np.savetxt(f'{prefix}confusion_matrix.csv', 
               metrics['confusion_matrix'], 
               delimiter=',', 
               fmt='%d')
    
    # Save per-class metrics
    with open(f'{prefix}classification_metrics.txt', 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(metrics['precision'])):
            f.write(f"{i:<10} {metrics['precision'][i]:<12.4f} {metrics['recall'][i]:<12.4f} {metrics['f1_score'][i]:<12.4f}\n")
        
        f.write(f"\nMacro Average:\n")
        f.write(f"Precision: {np.mean(metrics['precision']):.4f}\n")
        f.write(f"Recall: {np.mean(metrics['recall']):.4f}\n")
        f.write(f"F1-Score: {np.mean(metrics['f1_score']):.4f}\n")
    
    print(f"\nResults saved with prefix: {prefix}")
