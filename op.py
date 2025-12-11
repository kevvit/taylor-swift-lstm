import numpy as np

T = 1.2


def sigmoid(x, grad=False):
    if not grad:
        return 1 / (1 + np.exp(-x))
    return x * (1 - x)

def tanh(x, grad=False):
    if not grad:
        return np.tanh(x)
    return 1 - x**2

def softmax(x, axis):
    exps = np.exp((x - np.max(x, axis=axis, keepdims=True)) / T)
    return exps / np.sum(exps, axis=axis, keepdims=True)

def cross_entropy(prediction, target, reduction="mean"):
    eps = np.finfo(prediction.dtype).eps
    prediction = np.clip(prediction, eps, 1 - eps)

    loss = -np.take_along_axis(
        np.log(prediction), target[..., np.newaxis], axis=-1
    )

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
