"""
Softmax Regression Model Implementation from Scratch
Uses only NumPy for computations
"""
import numpy as np


class SoftmaxRegression:
    """
    Softmax Regression Classifier
    
    Parameters:
    -----------
    learning_rate : float
        Learning rate for gradient descent
    num_iterations : int
        Number of training iterations
    batch_size : int
        Size of mini-batches for training
    reg_lambda : float
        L2 regularization parameter
    """
    
    def __init__(self, learning_rate=0.001, num_iterations=1000, batch_size=128, reg_lambda=0.01, optimizer='adam'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer
        self.weights = None
        self.bias = None
        self.num_classes = None
        self.num_features = None
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
    
    def _softmax(self, z):
        """
        Compute softmax function
        
        Parameters:
        -----------
        z : numpy array of shape (num_samples, num_classes)
            Raw scores
            
        Returns:
        --------
        softmax probabilities of shape (num_samples, num_classes)
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_loss(self, X, y):
        """
        Compute cross-entropy loss with L2 regularization
        
        Parameters:
        -----------
        X : numpy array of shape (num_samples, num_features)
            Input features
        y : numpy array of shape (num_samples, num_classes)
            One-hot encoded labels
            
        Returns:
        --------
        loss : float
            Cross-entropy loss value
        """
        num_samples = X.shape[0]
        
        # Forward pass
        scores = np.dot(X, self.weights) + self.bias
        probs = self._softmax(scores)
        
        # Cross-entropy loss
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        cross_entropy_loss = -np.sum(y * np.log(probs + epsilon)) / num_samples
        
        # L2 regularization
        reg_loss = (self.reg_lambda / 2) * np.sum(self.weights ** 2)
        
        total_loss = cross_entropy_loss + reg_loss
        
        return total_loss
    
    def _compute_gradients(self, X, y, probs):
        """
        Compute gradients for weights and bias
        
        Parameters:
        -----------
        X : numpy array of shape (num_samples, num_features)
            Input features
        y : numpy array of shape (num_samples, num_classes)
            One-hot encoded labels
        probs : numpy array of shape (num_samples, num_classes)
            Predicted probabilities
            
        Returns:
        --------
        dW : gradient for weights
        db : gradient for bias
        """
        num_samples = X.shape[0]
        
        # Gradient of loss with respect to scores
        dscores = probs - y
        
        # Gradient with respect to weights (including L2 regularization)
        dW = np.dot(X.T, dscores) / num_samples + self.reg_lambda * self.weights
        
        # Gradient with respect to bias
        db = np.sum(dscores, axis=0) / num_samples
        
        return dW, db
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the softmax regression model
        
        Parameters:
        -----------
        X_train : numpy array of shape (num_samples, num_features)
            Training features
        y_train : numpy array of shape (num_samples,)
            Training labels (integers from 0 to num_classes-1)
        X_val : numpy array, optional
            Validation features
        y_val : numpy array, optional
            Validation labels
        verbose : bool
            Whether to print training progress
        """
        num_samples, self.num_features = X_train.shape
        self.num_classes = len(np.unique(y_train))
        
        # Initialize weights and bias
        np.random.seed(42)
        self.weights = np.random.randn(self.num_features, self.num_classes) * 0.01
        self.bias = np.zeros(self.num_classes)
        
        # Adam optimizer parameters
        if self.optimizer == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m_w, v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
            m_b, v_b = np.zeros_like(self.bias), np.zeros_like(self.bias)
            t = 0
        
        # Convert labels to one-hot encoding
        y_train_one_hot = np.eye(self.num_classes)[y_train]
        
        # Training loop
        current_lr = self.learning_rate
        
        for iteration in range(self.num_iterations):
            # Learning rate decay (StepLR equivalent)
            if iteration > 0 and iteration % 100 == 0:
                current_lr *= 0.95
                
            # Mini-batch gradient descent
            indices = np.random.permutation(num_samples)
            
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                y_batch = y_train_one_hot[batch_indices]
                
                # Forward pass
                scores = np.dot(X_batch, self.weights) + self.bias
                probs = self._softmax(scores)
                
                # Compute gradients
                dW, db = self._compute_gradients(X_batch, y_batch, probs)
                
                # Update parameters
                if self.optimizer == 'adam':
                    t += 1
                    # Update weights
                    m_w = beta1 * m_w + (1 - beta1) * dW
                    v_w = beta2 * v_w + (1 - beta2) * (dW ** 2)
                    m_w_hat = m_w / (1 - beta1 ** t)
                    v_w_hat = v_w / (1 - beta2 ** t)
                    self.weights -= current_lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                    
                    # Update bias
                    m_b = beta1 * m_b + (1 - beta1) * db
                    v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
                    m_b_hat = m_b / (1 - beta1 ** t)
                    v_b_hat = v_b / (1 - beta2 ** t)
                    self.bias -= current_lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
                else:
                    # Standard SGD
                    self.weights -= current_lr * dW
                    self.bias -= current_lr * db
            
            # Compute loss and accuracy every 10 iterations
            if iteration % 10 == 0 or iteration == self.num_iterations - 1:
                train_loss = self._compute_loss(X_train, y_train_one_hot)
                train_acc = self.score(X_train, y_train)
                
                self.training_history['loss'].append(train_loss)
                self.training_history['accuracy'].append(train_acc)
                
                if verbose:
                    if X_val is not None and y_val is not None:
                        val_acc = self.score(X_val, y_val)
                        print(f"Iteration {iteration}: Train Loss = {train_loss:.4f}, "
                              f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
                    else:
                        print(f"Iteration {iteration}: Train Loss = {train_loss:.4f}, "
                              f"Train Acc = {train_acc:.4f}")
        
        if verbose:
            print("\nTraining completed!")
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : numpy array of shape (num_samples, num_features)
            Input features
            
        Returns:
        --------
        probabilities : numpy array of shape (num_samples, num_classes)
            Predicted probabilities for each class
        """
        scores = np.dot(X, self.weights) + self.bias
        return self._softmax(scores)
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : numpy array of shape (num_samples, num_features)
            Input features
            
        Returns:
        --------
        predictions : numpy array of shape (num_samples,)
            Predicted class labels
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X, y):
        """
        Compute accuracy score
        
        Parameters:
        -----------
        X : numpy array of shape (num_samples, num_features)
            Input features
        y : numpy array of shape (num_samples,)
            True labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def save_model(self, filepath):
        """
        Save model parameters to file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        np.savez(filepath, 
                 weights=self.weights, 
                 bias=self.bias,
                 num_classes=self.num_classes,
                 num_features=self.num_features)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model parameters from file
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']
        self.num_classes = int(data['num_classes'])
        self.num_features = int(data['num_features'])
        print(f"Model loaded from {filepath}")
