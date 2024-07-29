import numpy as np

from conv import Conv2D
from pool import MaxPool2D
from linear import Linear
from activation import ReLU

from misc import Flatten
from loss import CrossEntropyLoss

import mnist

from model import Model


def evaluate(preds: np.array, labels: np.array) -> float:
    return np.mean((np.argmax(preds, axis=1) == labels))


def count_correct(preds: np.array, labels: np.array) -> int:
    return np.sum((np.argmax(preds, axis=1) == labels))


def train_log(epoch: int, epoch_count: int, loss: float, acc: float) -> None:
    loss_ = loss.round(4)
    acc_ = acc.round(4)
    print(f"Epoch: {epoch}/{epoch_count} Loss: {loss_} Acc: {acc_}")


def main():
    BATCH_SIZE = 30
    X_train_raw, y_train, X_test_raw, y_test = mnist.load_dataset()
    X_train = mnist.preprocess_dataset(X_train_raw)
    X_test = mnist.preprocess_dataset(X_test_raw)

    layers = [
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
    loss_fn = CrossEntropyLoss()
    epochs = 150
    lr = 1e-2

    image_count = X_train.shape[0]
    batch_count = image_count // BATCH_SIZE
    training_examples = X_train.shape[0]

    print(f"Number of training examples: {training_examples}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training examples per batch: {batch_count}")
    print(f"Epochs: {epochs}")

    for epoch in range(epochs):
        sample = np.random.randint(low=0, high=batch_count, size=(BATCH_SIZE,))
        X, y = X_train[sample], y_train[sample]

        for layer in layers:
            X = layer.forward(X)

        delta = loss_fn.gradient(X, y)

        if epoch % 10 == 0 and epoch > 0:
            loss = loss_fn.value(X, y)
            acc = evaluate(X, y)
            train_log(epoch, epochs, loss, acc)

        for layer in reversed(layers):
            delta = layer.backward(delta)

        for layer in layers:
            layer.update_weights(lr)

    print("Finished training.")

    print("Starting evaluation on test set.")

    test_correct = 0
    test_examples = X_test.shape[0]
    test_batch_count = test_examples // BATCH_SIZE

    for i in range(test_batch_count):
        batch_start, batch_end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        X = X_test[batch_start:batch_end]
        y = y_test[batch_start:batch_end]

        for layer in layers:
            X = layer.forward(X)

        test_correct += count_correct(X, y)

    print(f"Test results: {test_correct} / {test_examples} correct.")


if __name__ == "__main__":
    main()
