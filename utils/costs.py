# -*- coding: utf-8 -*-
"""A function to compute the cost."""

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