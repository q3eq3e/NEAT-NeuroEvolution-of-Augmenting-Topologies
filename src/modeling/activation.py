import numpy as np


def identity(x):
    return x


def sigmoid(x):
    # some times error occurs but after like 5 minutes
    return 1 / (1 + np.exp(-x))
