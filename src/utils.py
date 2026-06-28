import numpy as np

def scale(X, mean, std):
    return (X - mean) / std
