# -*- coding: utf-8 -*-
"""A function to compute the cost."""

from utils.helpers import sigmoid
import numpy as np

def mse(e):
    """compute the mse loss given an error vector.

    Args:
        e: the error vector, numpy array of shape (N,), N is the number of samples.
        
    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 N) in front of the sum
    """
    return e @ e / (2 * len(e))

def mae(e):
    """compute the mae loss given an error vector.

    Args:
        e: the error vector, numpy array of shape (N,), N is the number of samples.
        
    Returns:
        mae: scalar corresponding to the mae 
    """
    return np.mean(np.abs(e))

def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    >>> compute_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """

    e = y - tx @ w
    return mse(e)

def compute_mae(y, tx, w):
    """compute the loss by mae.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mae: scalar corresponding to the mae
    """

    e = y - tx.dot(w)
    return mae(e)


def neg_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(neg_log_likelihood(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    N = y.shape[0]

    # --- vectorized version ---
    pred = sigmoid(tx @ w)
    loss = y.T @ np.log(pred) + (1 - y).T @ np.log(1 - pred)
    return - 1 / N * loss.item()