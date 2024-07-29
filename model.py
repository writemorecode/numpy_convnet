import numpy as np

from conv import Conv2D
from pool import MaxPool2D
from linear import Linear
from activation import ReLU

from misc import Flatten
from loss import CrossEntropyLoss

from datetime import datetime


class Model:
    def __init__(self, loss_fn=CrossEntropyLoss, optimizer=None):
        self.loss_fn = loss_fn()
        self.optimizer = optimizer
        self.layers = [
            Conv2D(1, 6, 5),
            ReLU(),
            MaxPool2D(2),
            Conv2D(6, 16, 5),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Linear(400, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, 10),
        ]
        self.num_classes = 10

    def __repr__(self):
        layer_string = "Layers:\n" + "\n".join(map(str, self.layers))
        optimizer_string = f"Optimizer: {self.optimizer}"
        loss_string = f"Loss: {self.loss_fn}"
        return "\n".join([layer_string, optimizer_string, loss_string])

    def accuracy(self, preds: np.array, labels: np.array):
        return (np.argmax(preds, axis=1) == labels).mean()

    def forward(self, X: np.array) -> None:
        """
        Runs a minibatch forward through the network,
        and returns the output.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta: np.array) -> None:
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
