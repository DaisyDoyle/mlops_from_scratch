# tests/test_model.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import LogisticRegression

def test_predict_proba_range():
    """Output must always be a valid probability between 0 and 1."""
    model = LogisticRegression()
    model.weights = np.array([0.5, -0.3, 0.1, 0.8, -0.2, 0.4, -0.6])
    model.bias    = 0.1
    X = np.random.randn(100, 7)
    probs = model.predict_proba(X)
    assert probs.min() >= 0.0, "Probability below 0"
    assert probs.max() <= 1.0, "Probability above 1"

def test_gradient_descent_reduces_loss():
    """Loss must decrease over training iterations."""
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.001, n_iters=1000, lambda_=0.1)
    losses = model.fit(X, y)          
    assert losses[-1] < losses[0], "Loss did not decrease during training"

if __name__ == "__main__":
    test_predict_proba_range()
    test_gradient_descent_reduces_loss()
    print("All tests passed.")