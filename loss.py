import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def __repr__(self):
        return "CrossEntropyLoss"

    def softmax(self, X: np.array) -> np.array:
        X_max = np.max(X, axis=1, keepdims=True)
        X_exp = np.exp(X - X_max)
        X_exp_sum = np.sum(X_exp, axis=1, keepdims=True)
        Y = X_exp / X_exp_sum
        return Y

    def value(self, preds: np.array, labels: np.array) -> float:
        """
        labels: (B,) , preds: (B,N)
        """
        B = preds.shape[0]
        preds_softmax = self.softmax(preds)
        return -np.log(preds_softmax[range(B), labels]).mean()

    def gradient(self, preds: np.array, labels: np.array) -> np.array:
        """
        Returns the gradient of the cross entropy loss of 'preds' and 'labels'.
        Args:
            preds: (B,N)
            labels: (B,)
        """
        B = preds.shape[0]
        grad = preds.copy()
        grad[range(B), labels] -= 1
        return grad
