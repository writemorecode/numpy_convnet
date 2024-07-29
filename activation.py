import numpy as np


class ReLU:
    def __init__(self):
        pass

    def __repr__(self):
        return "ReLU"

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, grad_Y):
        return grad_Y * np.where(self.X >= 0, 1, 0)

    def update_weights(self, lr):
        pass
