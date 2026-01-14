import numpy as np


def identity(x: float) -> float:
    return x


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def tanh(x: float) -> float:
    return np.tanh(x)
