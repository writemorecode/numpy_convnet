import numpy as np


class Flatten:
    def __init__(self):
        pass

    def __repr__(self):
        return "Flatten"

    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, Y_grad):
        return Y_grad.reshape(self.shape)

    def update_weights(self, lr):
        pass
