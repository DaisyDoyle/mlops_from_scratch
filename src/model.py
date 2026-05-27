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
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def predict_proba(self, X):
        """Calculate predicted probabilities using the logistic function."""
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        return A

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train model using gradient descent.
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            A = self.predict_proba(X)
            self.losses.append(self.compute_loss(y,A))

            if X_val is not None and y_val is not None:
                A_val = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, A_val)
                
            dz = A - y 
            dw = (
                (1 / n_samples) * np.dot(X.T, dz) 
                + (self.lambda_ / n_samples) * self.weights
                )
            db = (1 / n_samples) * np.sum(dz)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        return self.losses
            
    def predict(self, X):
        """Predict binary class labels for samples in X."""
        threshold = .5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]
        
        return np.array(y_predicted_cls)