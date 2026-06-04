import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier built from scratch using gradient descent.
    Supports L2 regularization.
    """
    def __init__(self, learning_rate=0.001, n_iters=1000, lambda_=0.1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.val_losses = []
        self.losses = []
        self.lambda_ = lambda_  
         
    def _sigmoid(self, x):
        """Compute sigmoid activation."""
        x = np.clip(x, -500, 500) 
        return 1 / (1 + np.exp(-x)) # Clip input to prevent overflow, large +ve: sigmoid ~1, large -ve: sigmoid ~0, around 0: sigmoid ~0.5

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss.
        Function penalizes incorrect predictions more heavily as they become more confident (i.e. predicted probability approaches 0 for true class 1, or approaches 1 for true class 0).
        """
        epsilon = 1e-9 # Prevent log(0) 
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def predict_proba(self, X):
        """Calculate predicted probabilities using the logistic function."""
        z = np.dot(X, self.weights) + self.bias # Weighted sum of features plus bias, large +ve: confident class=1
        A = self._sigmoid(z) # Apply sigmoid to convert to probabilities, large -ve: confident class=0, around 0: uncertain
        return A

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train model using gradient descent.
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features) # Initialize weights to zero
        self.bias = 0

        for _ in range(self.n_iters):
            A = self.predict_proba(X) # Forward pass: compute predicted probabilities with current weights and bias
            self.losses.append(self.compute_loss(y,A)) # Measure incorrectness of predictions with current weights and bias, and store for plotting 

            if X_val is not None and y_val is not None:
                A_val = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, A_val)
                self.val_losses.append(val_loss) # New line - wasn't plotting validation loss before
                
            dz = A - y #  Compute error between predicted probabilities and true labels
            dw = (
                (1 / n_samples) * np.dot(X.T, dz) # Compute gradient of loss w.r.t. weights without regularization
                + (self.lambda_ / n_samples) * self.weights # Add L2 regularization term to weight gradients
                ) 
            db = (1 / n_samples) * np.sum(dz) # Compute gradient of loss w.r.t. bias
            
            self.weights -= self.lr * dw # Update weights by taking a step in the direction of the negative gradient, scaled by learning rate
            self.bias -= self.lr * db
            
        return self.losses
            
    def predict(self, X):
        """Predict binary class labels for samples in X.
        Threshold predicted probabilities at 0.5 to determine class labels: if predicted probability > 0.5, predict class 1; else predict class 0.
        """
        threshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]
        
        return np.array(y_predicted_cls)