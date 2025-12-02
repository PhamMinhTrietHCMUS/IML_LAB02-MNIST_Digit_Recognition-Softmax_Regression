"""
Feature Engineering Experiments for MNIST Classification
Compares different feature extraction methods to improve model performance
"""
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from softmax_regression import SoftmaxRegression
from data_loader import load_mnist_data, preprocess_images, split_train_val


class FeatureExtractor:
    """
    Implements different feature extraction methods
    """
    
    @staticmethod
    def extract_raw_pixels(images, normalize=True):
        """
        Feature 1: Normalized Pixel Intensity (Baseline)
        Simply flatten and normalize pixel values
        """
        if normalize:
            features = images.astype(np.float32) / 255.0
        else:
            features = images.astype(np.float32)
        
        # Flatten
        num_images = features.shape[0]
        features = features.reshape(num_images, -1)
        
        return features
    
    @staticmethod
    def extract_edge_features(images):
        """
        Feature 2: Edge Detection Features using Sobel-like Filters
        Detects horizontal and vertical edges without using external libraries
        """
        num_images = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Normalize images
        normalized = images.astype(np.float32) / 255.0
        
        features_list = []
        
        for i in range(num_images):
            img = normalized[i]
            
            # Horizontal gradient (Sobel-like)
            gx = np.zeros_like(img)
            gx[:, :-1] = np.diff(img, axis=1)  # Simple horizontal gradient
            
            # Vertical gradient (Sobel-like)
            gy = np.zeros_like(img)
            gy[:-1, :] = np.diff(img, axis=0)  # Simple vertical gradient
            
            # Gradient magnitude
            magnitude = np.sqrt(gx**2 + gy**2)
            
            # Gradient direction (quantized into 8 bins)
            direction = np.arctan2(gy, gx)
            
            # Combine: original pixels + edge magnitude + edge direction
            combined = np.stack([img, magnitude, np.sin(direction), np.cos(direction)], axis=0)
            
            # Flatten
            features_list.append(combined.flatten())
        
        return np.array(features_list)
    
    @staticmethod
    def extract_block_features(images, block_size=7):
        """
        Feature 3: Block Averaging Features (Dimensionality Reduction)
        Divides image into blocks and computes statistics
        """
        num_images = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Normalize images
        normalized = images.astype(np.float32) / 255.0
        
        # Calculate number of blocks
        n_blocks_h = height // block_size
        n_blocks_w = width // block_size
        
        features_list = []
        
        for i in range(num_images):
            img = normalized[i]
            block_features = []
            
            for bh in range(n_blocks_h):
                for bw in range(n_blocks_w):
                    # Extract block
                    block = img[
                        bh * block_size:(bh + 1) * block_size,
                        bw * block_size:(bw + 1) * block_size
                    ]
                    
                    # Compute statistics for this block
                    mean = np.mean(block)
                    std = np.std(block)
                    max_val = np.max(block)
                    min_val = np.min(block)
                    
                    # Add to features
                    block_features.extend([mean, std, max_val, min_val])
            
            features_list.append(block_features)
        
        return np.array(features_list)
    
    @staticmethod
    def extract_hog_features(images, cell_size=7, n_bins=9):
        """
        Feature 4: Histogram of Oriented Gradients (HOG) - Manual Implementation
        Captures edge and shape information
        """
        num_images = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Normalize images
        normalized = images.astype(np.float32) / 255.0
        
        # Calculate number of cells
        n_cells_h = height // cell_size
        n_cells_w = width // cell_size
        
        features_list = []
        
        for i in range(num_images):
            img = normalized[i]
            hog_features = []
            
            # Compute gradients
            gx = np.zeros_like(img)
            gy = np.zeros_like(img)
            gx[:, :-1] = np.diff(img, axis=1)
            gy[:-1, :] = np.diff(img, axis=0)
            
            # Gradient magnitude and orientation
            magnitude = np.sqrt(gx**2 + gy**2)
            orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
            
            # Divide into cells and compute histogram
            for ch in range(n_cells_h):
                for cw in range(n_cells_w):
                    # Extract cell
                    cell_mag = magnitude[
                        ch * cell_size:(ch + 1) * cell_size,
                        cw * cell_size:(cw + 1) * cell_size
                    ]
                    cell_ori = orientation[
                        ch * cell_size:(ch + 1) * cell_size,
                        cw * cell_size:(cw + 1) * cell_size
                    ]
                    
                    # Compute histogram of orientations
                    hist = np.zeros(n_bins)
                    bin_size = 180 / n_bins
                    
                    for y in range(cell_mag.shape[0]):
                        for x in range(cell_mag.shape[1]):
                            bin_idx = int(cell_ori[y, x] / bin_size)
                            if bin_idx >= n_bins:
                                bin_idx = n_bins - 1
                            hist[bin_idx] += cell_mag[y, x]
                    
                    # Normalize histogram
                    hist_norm = np.linalg.norm(hist)
                    if hist_norm > 0:
                        hist = hist / hist_norm
                    
                    hog_features.extend(hist)
            
            features_list.append(hog_features)
        
        return np.array(features_list)
    
    @staticmethod
    def extract_combined_features(images):
        """
        Feature 5: Combined Features
        Combines multiple feature types for best performance
        """
        # Extract different features
        raw = FeatureExtractor.extract_raw_pixels(images, normalize=True)
        block = FeatureExtractor.extract_block_features(images, block_size=7)
        
        # Concatenate
        combined = np.hstack([raw, block])
        
        return combined


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_name, learning_rate=0.1, num_iterations=500):
    """
    Train and evaluate model with given features
    """
    print(f"\n{'='*60}")
    print(f"Training with {feature_name}")
    print(f"{'='*60}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Split into train/val
    X_train_split, X_val, y_train_split, y_val = split_train_val(
        X_train, y_train, val_ratio=0.1, random_state=42
    )
    
    # Train model
    model = SoftmaxRegression(
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        batch_size=128,
        reg_lambda=0.001
    )
    
    model.fit(X_train_split, y_train_split, X_val, y_val, verbose=False)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return model, train_acc, test_acc


def main():
    """
    Main function to compare different feature designs
    """
    print("="*60)
    print("Feature Engineering Experiments for MNIST Classification")
    print("="*60)
    
    # Load data
    print("\nLoading MNIST dataset...")
    X_train_raw, y_train, X_test_raw, y_test = load_mnist_data(data_dir='archive_2')
    
    print(f"Training samples: {X_train_raw.shape[0]}")
    print(f"Test samples: {X_test_raw.shape[0]}")
    
    # Dictionary to store results
    results = {}
    
    # Experiment 1: Normalized Pixel Intensity (Baseline)
    print("\n" + "="*60)
    print("EXPERIMENT 1: Normalized Pixel Intensity Features")
    print("="*60)
    print("Description: Raw pixel values normalized to [0, 1] range")
    print("Dimensionality: 28×28 = 784 features")
    
    X_train_1 = FeatureExtractor.extract_raw_pixels(X_train_raw, normalize=True)
    X_test_1 = FeatureExtractor.extract_raw_pixels(X_test_raw, normalize=True)
    
    model_1, train_acc_1, test_acc_1 = train_and_evaluate(
        X_train_1, y_train, X_test_1, y_test, 
        "Normalized Pixel Intensity"
    )
    results['Normalized Pixels'] = (train_acc_1, test_acc_1, X_train_1.shape[1])
    
    # Save this model (baseline)
    os.makedirs('models', exist_ok=True)
    model_1.save_model('models/softmax_model.npz')
    
    # Experiment 2: Edge Detection Features
    print("\n" + "="*60)
    print("EXPERIMENT 2: Edge Detection Features")
    print("="*60)
    print("Description: Sobel-like edge detection (horizontal/vertical gradients)")
    print("Features: Original pixels + gradient magnitude + gradient direction")
    
    X_train_2 = FeatureExtractor.extract_edge_features(X_train_raw)
    X_test_2 = FeatureExtractor.extract_edge_features(X_test_raw)
    
    model_2, train_acc_2, test_acc_2 = train_and_evaluate(
        X_train_2, y_train, X_test_2, y_test,
        "Edge Detection Features"
    )
    results['Edge Features'] = (train_acc_2, test_acc_2, X_train_2.shape[1])
    
    # Experiment 3: Block Averaging (Dimensionality Reduction)
    print("\n" + "="*60)
    print("EXPERIMENT 3: Block Averaging Features")
    print("="*60)
    print("Description: Divide image into 7×7 blocks, compute statistics per block")
    print("Features: Mean, Std, Max, Min for each block (4 features per block)")
    
    X_train_3 = FeatureExtractor.extract_block_features(X_train_raw, block_size=7)
    X_test_3 = FeatureExtractor.extract_block_features(X_test_raw, block_size=7)
    
    model_3, train_acc_3, test_acc_3 = train_and_evaluate(
        X_train_3, y_train, X_test_3, y_test,
        "Block Averaging Features"
    )
    results['Block Features'] = (train_acc_3, test_acc_3, X_train_3.shape[1])
    
    # Experiment 4: HOG Features
    print("\n" + "="*60)
    print("EXPERIMENT 4: Histogram of Oriented Gradients (HOG)")
    print("="*60)
    print("Description: Gradient histograms in local cells")
    print("Features: 9 orientation bins per 7×7 cell")
    
    X_train_4 = FeatureExtractor.extract_hog_features(X_train_raw, cell_size=7, n_bins=9)
    X_test_4 = FeatureExtractor.extract_hog_features(X_test_raw, cell_size=7, n_bins=9)
    
    model_4, train_acc_4, test_acc_4 = train_and_evaluate(
        X_train_4, y_train, X_test_4, y_test,
        "HOG Features"
    )
    results['HOG Features'] = (train_acc_4, test_acc_4, X_train_4.shape[1])
    
    # Experiment 5: Combined Features
    print("\n" + "="*60)
    print("EXPERIMENT 5: Combined Features")
    print("="*60)
    print("Description: Concatenation of normalized pixels + block features")
    
    X_train_5 = FeatureExtractor.extract_combined_features(X_train_raw)
    X_test_5 = FeatureExtractor.extract_combined_features(X_test_raw)
    
    model_5, train_acc_5, test_acc_5 = train_and_evaluate(
        X_train_5, y_train, X_test_5, y_test,
        "Combined Features"
    )
    results['Combined Features'] = (train_acc_5, test_acc_5, X_train_5.shape[1])
    
    # Summary Report
    print("\n" + "="*60)
    print("FINAL COMPARISON OF ALL FEATURE DESIGNS")
    print("="*60)
    print(f"\n{'Feature Design':<25} {'Dims':<10} {'Train Acc':<12} {'Test Acc':<12} {'Improvement'}")
    print("-"*80)
    
    baseline_test_acc = results['Normalized Pixels'][1]
    
    for name, (train_acc, test_acc, dims) in results.items():
        improvement = (test_acc - baseline_test_acc) * 100
        improvement_str = f"{improvement:+.2f}%" if name != 'Normalized Pixels' else "baseline"
        print(f"{name:<25} {dims:<10} {train_acc*100:>6.2f}%     {test_acc*100:>6.2f}%     {improvement_str}")
    
    # Find best model
    best_name = max(results, key=lambda k: results[k][1])
    best_test_acc = results[best_name][1]
    
    print("\n" + "="*60)
    print(f"BEST PERFORMING MODEL: {best_name}")
    print(f"Test Accuracy: {best_test_acc*100:.2f}%")
    print("="*60)
    
    # Save summary to file
    with open('models/feature_comparison.txt', 'w') as f:
        f.write("Feature Engineering Comparison Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Feature Design':<25} {'Dims':<10} {'Train Acc':<12} {'Test Acc':<12} {'Improvement'}\n")
        f.write("-"*80 + "\n")
        
        for name, (train_acc, test_acc, dims) in results.items():
            improvement = (test_acc - baseline_test_acc) * 100
            improvement_str = f"{improvement:+.2f}%" if name != 'Normalized Pixels' else "baseline"
            f.write(f"{name:<25} {dims:<10} {train_acc*100:>6.2f}%     {test_acc*100:>6.2f}%     {improvement_str}\n")
        
        f.write(f"\nBest Model: {best_name}\n")
        f.write(f"Best Test Accuracy: {best_test_acc*100:.2f}%\n")
    
    print("\nResults saved to models/feature_comparison.txt")


if __name__ == '__main__':
    main()
