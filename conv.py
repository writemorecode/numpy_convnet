import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    """
    Input shape: (N,C_in,W,W)
    Output shape: (N,C_out,W-K+1,W-K+1)
    Padding: 0, Stride: 1, Dilation: 0
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size

        bound = np.sqrt(1.0 / (self.C_in * self.K * self.K))

        self.W = np.random.uniform(
            -bound, bound, size=(self.C_out, self.C_in, self.K, self.K)
        )
        self.b = np.random.uniform(-bound, bound, size=(self.C_out))

        self.W_grad = None
        self.b_grad = None

    def __repr__(self):
        return f"Conv2D: C_in={self.C_in}, C_out={self.C_out}, K={self.K}"

    def forward(self, X):
        X_windows = sliding_window_view(X, window_shape=(self.K, self.K), axis=(2, 3))
        self.cache = X_windows

        Y = np.einsum("bihwkl,oikl->bohw", X_windows, self.W)
        Y += self.b[None, :, None, None]
        return Y

    def backward(self, Y_grad):
        X_windows = self.cache

        pad = self.K - 1
        Y_grad_padded = np.pad(Y_grad, ((0,), (0,), (pad,), (pad,)))

        Y_grad_windows = sliding_window_view(
            Y_grad_padded, window_shape=(self.K, self.K), axis=(2, 3)
        )
        W_rotated = np.rot90(self.W, 2, axes=(2, 3))

        self.b_grad = np.sum(Y_grad, axis=(0, 2, 3))
        self.W_grad = np.einsum("bihwkl,bohw->oikl", X_windows, Y_grad)
        X_grad = np.einsum("bohwkl,oikl->bihw", Y_grad_windows, W_rotated)

        return X_grad

    def update_weights(self, lr):
        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad
