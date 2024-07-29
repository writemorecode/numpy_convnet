import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.F_in = in_features
        self.F_out = out_features
        # X: (B, F_in)
        # W: (F_in, F_out), b: (F_out,)
        # Y = XW + b : (B,F_out)

        bound = np.sqrt(1 / self.F_in)
        self.W = np.random.uniform(-bound, bound, size=(self.F_in, self.F_out))
        self.b = np.random.uniform(-bound, bound, size=(self.F_out))

        self.W_grad = None
        self.b_grad = None

    def __repr__(self):
        return f"Linear: F_in={self.F_in}, F_out={self.F_out}"

    def forward(self, X):
        self.cache = X
        Y = X @ self.W + self.b
        return Y

    def backward(self, Y_grad):
        X = self.cache

        self.W_grad = X.T @ Y_grad
        self.b_grad = np.sum(Y_grad, axis=0)
        grad_X = Y_grad @ self.W.T

        return grad_X

    def update_weights(self, lr):
        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad
