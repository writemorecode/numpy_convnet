import numpy as np


class MaxPool2D:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def __repr__(self):
        return f"MaxPool2D: P={self.kernel_size}, stride={self.stride}"

    def forward(self, X):
        if len(X.shape) == 2:
            X = X.reshape(1, 1, *X.shape)
        N, C, H, W = X.shape
        P = self.kernel_size
        X_pools = X.reshape(N, C, H // P, P, W // P, P)
        X_pool_max = X_pools.max(axis=3).max(axis=4)
        self.cache = X, X_pools, X_pool_max
        return X_pool_max

    def backward(self, dout):
        X, X_reshaped, out = self.cache

        dx_reshaped = np.zeros_like(X_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = X_reshaped == out_newaxis
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(X.shape)
        return dx

    def update_weights(self, lr):
        pass
