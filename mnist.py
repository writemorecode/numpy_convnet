import numpy as np
import os
import struct


DATA_DIRECTORY = "data/"
TRAIN_IMG_FILE = "train-images.idx3-ubyte"
TRAIN_LABEL_FILE = "train-labels.idx1-ubyte"
TEST_IMG_FILE = "t10k-images.idx3-ubyte"
TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"


def load_images(path):
    with open(path, "rb") as fh:
        header = fh.read(4 * 4)
        magic, count, rows, cols = struct.unpack(">4I", header)
        assert magic == 0x803, f"Invalid MNIST magic: got {hex(magic)} expected 0x803."
        data = np.fromfile(fh, dtype=np.uint8).reshape(count, rows, cols)
    return data


def load_labels(path):
    with open(path, "rb") as fh:
        header = fh.read(4 * 2)
        magic, count = struct.unpack(">2I", header)
        assert magic == 0x801, f"Invalid MNIST magic: got {hex(magic)} expected 0x801."
        data = np.fromfile(fh, dtype=np.uint8)
    return data


def load_dataset():
    X_train = load_images(os.path.join(DATA_DIRECTORY, TRAIN_IMG_FILE))
    X_test = load_images(os.path.join(DATA_DIRECTORY, TEST_IMG_FILE))
    y_train = load_labels(os.path.join(DATA_DIRECTORY, TRAIN_LABEL_FILE))
    y_test = load_labels(os.path.join(DATA_DIRECTORY, TEST_LABEL_FILE))
    return X_train, y_train, X_test, y_test


def preprocess_dataset(X):
    X = X.astype(np.float32) / 255
    old_dim, new_dim = 28, 32
    pad = (new_dim - old_dim) // 2
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))
    X = X[:, np.newaxis, :, :]
    X -= X.mean()
    return X
