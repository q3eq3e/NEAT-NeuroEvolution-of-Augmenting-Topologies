import numpy as np


def identity(x):
    return x


def sigmoid(x):
    # some times error occurs but after like 5 minutes
    # if abs(x) > 30:
    #     raise ValueError("very weird input for sigmoid")
    return 1 / (1 + np.exp(-x))
