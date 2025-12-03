"""
MNIST Data Loader and Preprocessing
Reads IDX file format and provides preprocessing utilities
"""
import numpy as np
import struct
import os


def read_idx_labels(filename):
    """
    Read MNIST labels from IDX1 file format
    
    Parameters:
    -----------
    filename : str
        Path to the IDX1 labels file
        
    Returns:
    --------
    labels : numpy array
        Array of labels
    """
    with open(filename, 'rb') as f:
        # Read magic number and number of items
        magic, num_items = struct.unpack('>II', f.read(8))
        
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in label file {filename}')
        
        # Read labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels


def read_idx_images(filename):
    """
    Read MNIST images from IDX3 file format
    
    Parameters:
    -----------
    filename : str
        Path to the IDX3 images file
        
    Returns:
    --------
    images : numpy array of shape (num_images, rows, cols)
        Array of images
    """
    with open(filename, 'rb') as f:
        # Read magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in image file {filename}')
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
    return images


def load_mnist_data(data_dir='archive_2'):
    """
    Load MNIST dataset from IDX files
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MNIST IDX files
        
    Returns:
    --------
    X_train : numpy array of shape (num_train, 28, 28)
        Training images
    y_train : numpy array of shape (num_train,)
        Training labels
    X_test : numpy array of shape (num_test, 28, 28)
        Test images
    y_test : numpy array of shape (num_test,)
        Test labels
    """
    # File paths
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    
    # Check if files exist
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    print("Loading MNIST dataset...")
    X_train = read_idx_images(train_images_path)
    y_train = read_idx_labels(train_labels_path)
    X_test = read_idx_images(test_images_path)
    y_test = read_idx_labels(test_labels_path)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Image shape: {X_train.shape[1]}x{X_train.shape[2]}")
    
    return X_train, y_train, X_test, y_test


def normalize_images(images):
    """
    Normalize image pixel values to [0, 1] range
    
    Parameters:
    -----------
    images : numpy array
        Images with pixel values in [0, 255]
        
    Returns:
    --------
    normalized_images : numpy array
        Images with pixel values in [0, 1]
    """
    return images.astype(np.float32) / 255.0


def flatten_images(images):
    """
    Flatten images from (num_images, rows, cols) to (num_images, rows*cols)
    
    Parameters:
    -----------
    images : numpy array of shape (num_images, rows, cols)
        Images to flatten
        
    Returns:
    --------
    flattened_images : numpy array of shape (num_images, rows*cols)
        Flattened images
    """
    num_images = images.shape[0]
    return images.reshape(num_images, -1)


def preprocess_images(images, normalize=True, flatten=True):
    """
    Preprocess images: normalize and flatten
    
    Parameters:
    -----------
    images : numpy array
        Raw images
    normalize : bool
        Whether to normalize pixel values
    flatten : bool
        Whether to flatten images
        
    Returns:
    --------
    processed_images : numpy array
        Preprocessed images
    """
    processed = images.copy()
    
    if normalize:
        processed = normalize_images(processed)
    
    if flatten:
        processed = flatten_images(processed)
    
    return processed


def add_bias_feature(X):
    """
    Add bias feature (column of ones) to feature matrix
    
    Parameters:
    -----------
    X : numpy array of shape (num_samples, num_features)
        Feature matrix
        
    Returns:
    --------
    X_with_bias : numpy array of shape (num_samples, num_features + 1)
        Feature matrix with bias column
    """
    num_samples = X.shape[0]
    return np.column_stack([X, np.ones(num_samples)])


def extract_hog_features(images):
    """
    Extract HOG (Histogram of Oriented Gradients) features
    This is a simple implementation for feature engineering
    
    Parameters:
    -----------
    images : numpy array of shape (num_images, rows, cols)
        Input images
        
    Returns:
    --------
    features : numpy array
        HOG features
    """
    # For simplicity, we'll compute gradient-based features
    num_images = images.shape[0]
    features_list = []
    
    for i in range(num_images):
        img = images[i].astype(np.float32)
        
        # Compute gradients
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        
        gx[:, :-1] = np.diff(img, axis=1)
        gy[:-1, :] = np.diff(img, axis=0)
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Flatten and concatenate
        features_list.append(magnitude.flatten())
    
    return np.array(features_list)


def create_polynomial_features(X, degree=2):
    """
    Create polynomial features
    
    Parameters:
    -----------
    X : numpy array of shape (num_samples, num_features)
        Original features
    degree : int
        Polynomial degree
        
    Returns:
    --------
    poly_features : numpy array
        Polynomial features
    """
    if degree == 1:
        return X
    
    # For degree=2, add squared features
    if degree == 2:
        X_squared = X ** 2
        return np.column_stack([X, X_squared])
    
    return X


def split_train_val(X, y, val_ratio=0.1, random_state=42):
    """
    Split training data into train and validation sets
    
    Parameters:
    -----------
    X : numpy array
        Features
    y : numpy array
        Labels
    val_ratio : float
        Ratio of validation set
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_val, y_train, y_val
    """
    np.random.seed(random_state)
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    return X_train, X_val, y_train, y_val


def preprocess_single_image(image, target_size=(28, 28)):
    """
    Preprocess a single image for prediction
    Used for web app input
    
    Parameters:
    -----------
    image : numpy array
        Input image (can be any size)
    target_size : tuple
        Target size for the image
        
    Returns:
    --------
    preprocessed : numpy array of shape (1, num_features)
        Preprocessed image ready for prediction
    """
    from PIL import Image, ImageFilter
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype(np.uint8))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert to numpy array to check inversion
    img_array = np.array(image)
    
    # Invert colors if needed (MNIST has white digits on black background)
    # Check if the image has more white than black
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        image = Image.fromarray(img_array)
    
    # --- ADVANCED PREPROCESSING (MNIST Style with thin pen support) ---
    img_array = np.array(image)
    
    # 1. Find bounding box of the digit
    coords = cv2_get_bbox(img_array)
    
    if coords:
        x, y, w, h = coords
        # Crop the digit
        digit = image.crop((x, y, x+w, y+h))
        
        # 2. Resize to fit in 20x20 box (keeping aspect ratio)
        width, height = digit.size
        if width > height:
            new_w = 20
            new_h = max(1, int(height * (20 / width)))
        else:
            new_h = 20
            new_w = max(1, int(width * (20 / height)))
        
        digit = digit.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # --- MINIMAL PREPROCESSING (Keep original shape) ---
        digit_arr = np.array(digit).astype(np.uint8)
        
        # 3. Paste into 28x28 using Center of Mass (with limited shift)
        new_image = Image.new('L', target_size, 0)
        
        # Calculate geometric center position first
        geo_paste_x = (target_size[0] - new_w) // 2
        geo_paste_y = (target_size[1] - new_h) // 2
        
        # Create temp canvas to calculate CoM
        temp_canvas = Image.new('L', target_size, 0)
        temp_canvas.paste(digit, (geo_paste_x, geo_paste_y))
        temp_arr = np.array(temp_canvas)
        
        # Calculate Center of Mass
        cy, cx = get_center_of_mass(temp_arr)
        
        # Calculate shift (limit to max 2 pixels to prevent distortion)
        shift_x = np.clip(14 - cx, -2, 2)
        shift_y = np.clip(14 - cy, -2, 2)
        
        # Apply minimal shift
        final_paste_x = int(geo_paste_x + shift_x)
        final_paste_y = int(geo_paste_y + shift_y)
        
        new_image.paste(digit, (final_paste_x, final_paste_y))
        
        image = new_image
        img_array = np.array(image)
    else:
        # Fallback if no digit found (empty canvas)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(image)
    
    # Normalize and flatten
    img_normalized = img_array.astype(np.float32) / 255.0
    img_flattened = img_normalized.reshape(1, -1)
    
    return img_flattened


def get_center_of_mass(img):
    """
    Calculate center of mass of an image
    """
    total_mass = np.sum(img)
    if total_mass == 0:
        return 14, 14
    
    rows, cols = img.shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    cy = np.sum(y_coords * img) / total_mass
    cx = np.sum(x_coords * img) / total_mass
    
    return cy, cx


def cv2_get_bbox(img_array):
    """
    Helper to find bounding box of digit using simple thresholding
    Simulates cv2.boundingRect without needing opencv
    """
    # Threshold to find non-zero pixels
    rows = np.any(img_array > 50, axis=1)
    cols = np.any(img_array > 50, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Add a small padding
    h, w = img_array.shape
    ymin = max(0, ymin - 2)
    ymax = min(h, ymax + 2)
    xmin = max(0, xmin - 2)
    xmax = min(w, xmax + 2)
    
    return xmin, ymin, xmax-xmin, ymax-ymin
